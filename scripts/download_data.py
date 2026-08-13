import argparse
import hashlib
import io
import json
import os
import subprocess
from itertools import islice

import numpy as np
import soundfile as sf

from src.common.config import CONFIG, DATA, ENVS, ROOT, ensure_layout
from src.common.registry import data_entries, validate_registry
from src.common.results import write_json


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def registry_digest():
    value = hashlib.sha256()
    for name in ["data.json", "tts_texts.json"]:
        value.update(name.encode())
        value.update((CONFIG / name).read_bytes())
    return value.hexdigest()


def cached_data_valid():
    asr_path = DATA / "asr" / "manifest.json"
    tts_path = DATA / "tts" / "manifest.json"
    if not asr_path.is_file() or not tts_path.is_file():
        return False
    try:
        asr = json.loads(asr_path.read_text(encoding="utf-8"))
        tts = json.loads(tts_path.read_text(encoding="utf-8"))
        expected = registry_digest()
        if asr.get("registry_sha256") != expected or tts.get("registry_sha256") != expected:
            return False
        if {item["id"] for item in asr["samples"]} != {item["id"] for item in data_entries()}:
            return False
        for item in asr["samples"]:
            path = DATA.parent / item["path"]
            if not path.is_file() or digest(path) != item["sha256"]:
                return False
            augmentation = item.get("augmentation", {})
            if augmentation:
                noise = DATA.parent / augmentation["noise_path"]
                if not noise.is_file() or digest(noise) != augmentation["noise_sha256"]:
                    return False
        reference = DATA.parent / tts["reference_voice"]
        return reference.is_file() and digest(reference) == tts["reference_voice_sha256"]
    except (KeyError, json.JSONDecodeError, OSError):
        return False


def decode(value):
    if value.get("bytes"):
        audio, rate = sf.read(io.BytesIO(value["bytes"]), dtype="float32")
    else:
        audio, rate = sf.read(value["path"], dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != 16000:
        from scipy.signal import resample_poly

        audio = resample_poly(audio, 16000, rate).astype(np.float32)
        rate = 16000
    return audio.astype(np.float32), rate


def stream(entry):
    from datasets import Audio, load_dataset

    arguments = {
        "path": entry["source"],
        "split": entry["split"],
        "streaming": True,
        "revision": entry["revision"],
    }
    if entry.get("config"):
        arguments["name"] = entry["config"]
    dataset = load_dataset(**arguments)
    return dataset.cast_column("audio", Audio(decode=False))


def select(entry):
    selector = entry["selector"]
    items = list(islice(stream(entry), selector.get("scan", 1)))
    if not items:
        raise RuntimeError(f"no rows for {entry['id']}")
    kind = selector["type"]
    if kind == "first":
        return items[0]
    if kind == "index":
        return items[selector["value"]]
    if kind == "contains_any":
        values = selector["values"]
        field = entry["transcript_field"]
        return next(
            (item for item in items if any(value in item[field].lower() for value in values)),
            items[0],
        )
    if kind == "field_equals":
        return next(
            item
            for item in items
            if str(item.get(selector["field"], "")).lower() == selector["value"].lower()
        )
    if kind == "max_speech_rate":
        field = entry["transcript_field"]
        candidates = []
        for item in items:
            audio, rate = decode(item["audio"])
            duration = len(audio) / rate
            if selector["min_duration_s"] <= duration <= selector["max_duration_s"]:
                candidates.append((len(item[field].split()) / duration, item))
        return max(candidates, key=lambda pair: pair[0])[1]
    raise ValueError(f"invalid selector {kind}")


def save(entry, item):
    audio, rate = decode(item["audio"])
    path = DATA / "asr" / f"{entry['id']}.wav"
    sf.write(path, audio, rate, subtype="PCM_16")
    return {
        "id": entry["id"],
        "category": entry["category"],
        "path": str(path.relative_to(DATA.parent)),
        "sha256": digest(path),
        "transcript": item[entry["transcript_field"]].strip(),
        "duration_s": len(audio) / rate,
        "source": f"{entry['source']} {entry['split']}",
        "revision": entry["revision"],
        "source_id": str(item.get("id", item.get("speaker_id", ""))),
        "license": entry["license"],
        "source_metadata": {
            key: item[key]
            for key in ["speaker_id", "accent", "region"]
            if key in item and isinstance(item[key], (str, int, float))
        },
    }


def concatenate(entry):
    selector = entry["selector"]
    parts = []
    texts = []
    ids = []
    duration = 0.0
    silence = np.zeros(round(selector["silence_s"] * 16000), dtype=np.float32)
    for item in stream(entry):
        audio, rate = decode(item["audio"])
        parts.extend([audio, silence])
        texts.append(item[entry["transcript_field"]].strip())
        ids.append(str(item.get("id", "")))
        duration += len(audio) / rate + selector["silence_s"]
        if duration >= selector["target_duration_s"]:
            break
    audio = np.concatenate(parts)
    path = DATA / "asr" / f"{entry['id']}.wav"
    sf.write(path, audio, 16000, subtype="PCM_16")
    return {
        "id": entry["id"],
        "category": entry["category"],
        "path": str(path.relative_to(DATA.parent)),
        "sha256": digest(path),
        "transcript": " ".join(texts),
        "duration_s": len(audio) / 16000,
        "source": f"{entry['source']} {entry['split']}",
        "revision": entry["revision"],
        "source_id": ids,
        "license": entry["license"],
        "construction": selector,
    }


def transform(entry, source):
    audio, rate = sf.read(DATA.parent / source["path"], dtype="float32")
    spec = entry["transform"]
    rng = np.random.default_rng(spec["seed"])
    noise = np.cumsum(rng.normal(0, 1, len(audio))).astype(np.float64)
    noise -= noise.mean()
    noise /= max(np.max(np.abs(noise)), 1e-9)
    signal_rms = np.sqrt(np.mean(audio**2))
    noise_rms = np.sqrt(np.mean(noise**2))
    noise *= signal_rms / (noise_rms * 10 ** (spec["snr_db"] / 20))
    mixed = np.clip(audio + noise, -1, 1).astype(np.float32)
    noise_path = DATA / "asr" / f"{entry['id']}_noise.wav"
    path = DATA / "asr" / f"{entry['id']}.wav"
    sf.write(noise_path, noise.astype(np.float32), rate, subtype="PCM_16")
    sf.write(path, mixed, rate, subtype="PCM_16")
    return {
        "id": entry["id"],
        "category": entry["category"],
        "path": str(path.relative_to(DATA.parent)),
        "sha256": digest(path),
        "transcript": source["transcript"],
        "duration_s": source["duration_s"],
        "source": source["source"],
        "revision": source["revision"],
        "source_id": source["source_id"],
        "license": source["license"],
        "augmentation": {
            **spec,
            "noise_path": str(noise_path.relative_to(DATA.parent)),
            "noise_sha256": digest(noise_path),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--internal", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    validate_registry()
    entries = data_entries()
    if args.dry_run:
        print(json.dumps(entries, indent=2))
        return
    if not args.internal and not args.reference_only and not args.force and cached_data_valid():
        print("data already verified")
        return
    if not args.internal:
        env = os.environ.copy()
        env["UV_CACHE_DIR"] = str(ROOT / ".uv-cache")
        env["UV_PROJECT_ENVIRONMENT"] = str(ENVS / "data")
        command = [
            "uv",
            "run",
            "--frozen",
            "--no-default-groups",
            "--group",
            "data",
            "python",
            __file__,
            "--internal",
        ]
        if args.reference_only:
            command.append("--reference-only")
        if args.force:
            command.append("--force")
        subprocess.run(
            command,
            check=True,
            env=env,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        return
    from src.common.console import console, progress

    ensure_layout()
    if args.reference_only:
        manifest = json.loads((DATA / "asr" / "manifest.json").read_text(encoding="utf-8"))
        samples = {item["id"]: item for item in manifest["samples"]}
        write_reference(samples)
        return
    samples = {}
    with progress() as display:
        task = display.add_task("data", total=len(entries))
        for entry in entries:
            display.update(task, description=f"data {entry['id']}")
            if "derived_from" in entry:
                samples[entry["id"]] = transform(entry, samples[entry["derived_from"]])
            elif entry["selector"]["type"] == "concatenate":
                samples[entry["id"]] = concatenate(entry)
            else:
                samples[entry["id"]] = save(entry, select(entry))
            display.advance(task)
    write_reference(samples)
    write_json(
        DATA / "asr" / "manifest.json",
        {"registry_sha256": registry_digest(), "samples": list(samples.values())},
    )
    console.print(f"saved {DATA / 'asr' / 'manifest.json'}")


def write_reference(samples):
    from src.common.console import console

    config = json.loads((CONFIG / "data.json").read_text(encoding="utf-8"))
    source = samples[config["tts_reference_source"]]
    clean_audio, clean_rate = sf.read(DATA.parent / source["path"], dtype="float32")
    reference_path = DATA / "tts" / "reference_voice.wav"
    sf.write(
        reference_path,
        clean_audio[: clean_rate * 20],
        clean_rate,
        subtype="PCM_16",
    )
    write_json(
        DATA / "tts" / "manifest.json",
        {
            "registry_sha256": registry_digest(),
            "texts": json.loads((CONFIG / "tts_texts.json").read_text(encoding="utf-8")),
            "reference_voice": "data/tts/reference_voice.wav",
            "reference_source": source["id"],
            "reference_voice_sha256": digest(reference_path),
        },
    )
    console.print(f"saved {reference_path}")


if __name__ == "__main__":
    main()
