import argparse
import importlib
import json
import re
import time
import traceback

import psutil
import soundfile as sf

from src.common.config import CONFIG, DATA, OUTPUTS, benchmark_config, model_dir, models_manifest
from src.common.metrics import audio_metrics
from src.common.results import repetition_stats, write_json
from src.common.system import (
    ResourceSampler,
    directory_size,
    environment,
    set_seed,
    set_threads,
    suppress_native_dialogs,
)

MODULES = {
    name: f"src.tts.{name}" for name in ["pocket", "kokoro", "chatterbox", "kitten", "piper"]
}


def file_token(value):
    if value is None:
        return "default"
    token = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value)).strip("_").lower()
    return token or "custom"


def render(module, model, text, voice, speed, reference, interval, controls=None):
    sampler = ResourceSampler(interval).start()
    started = time.perf_counter()
    audio, rate, first = module.generate(
        model,
        text,
        voice=voice,
        speed=speed,
        reference=reference,
        **(controls or {}),
    )
    total = time.perf_counter() - started
    resources = sampler.stop()
    quality = audio_metrics(audio, rate)
    timing = {
        "first_audio_s": first,
        "first_playable_chunk_s": first,
        "total_s": total,
        "audio_duration_s": quality["duration_s"],
        "rtf": total / quality["duration_s"] if quality["duration_s"] else None,
        "samples_per_second": len(audio) / total if total else None,
    }
    return audio, rate, timing, resources, quality


def capability_rows(module, results):
    rows = []
    for feature in [
        "native_streaming",
        "voice_cloning",
        "multiple_voices",
        "speed",
        "paralinguistic_tags",
        "pronunciation_control",
        "text_normalization",
        "noise_control",
        "phoneme_width",
        "browser",
    ]:
        supported = module.CAPABILITIES.get(feature, False)
        tested = any(
            item["feature"] == feature and item["status"] != "unsupported" for item in results
        )
        rows.append(
            {
                "feature": feature,
                "supported": supported,
                "tested": tested,
                "result": next((item for item in results if item["feature"] == feature), {}),
            }
        )
    return rows


def special_tests(name, module, model, reference, interval, output, smoke=False):
    specs = json.loads((CONFIG / "special_tests.json").read_text(encoding="utf-8"))[name]
    values = []
    for spec in specs:
        feature = spec["feature"]
        if not module.CAPABILITIES.get(feature, False):
            values.append({"feature": feature, "status": "unsupported"})
            continue
        variants = spec.get("values", [None])
        if smoke:
            variants = variants[:1]
        for variant in variants:
            voice = models_manifest()["tts"][name]["voice"]
            speed = 1.0
            prompt = None
            controls = {}
            if feature == "speed":
                speed = variant
            elif feature == "multiple_voices":
                voice = variant
            elif feature == "voice_cloning":
                prompt = reference
            elif feature == "noise_control":
                controls["noise_scale"] = variant
            elif feature == "phoneme_width":
                controls["noise_w_scale"] = variant
            elif feature == "text_normalization":
                controls["clean_text"] = variant
            elif feature == "pronunciation_control":
                controls["phonemes"] = variant
            try:
                audio, rate, timing, resources, quality = render(
                    module,
                    model,
                    spec["text"],
                    voice,
                    speed,
                    prompt,
                    interval,
                    controls,
                )
            except Exception as error:
                message = str(error)
                gated = feature == "voice_cloning" and "accept the terms" in message.lower()
                values.append(
                    {
                        "feature": feature,
                        "status": "gated" if gated else "failed",
                        "variant": variant,
                        "error_type": type(error).__name__,
                        "error": message,
                    }
                )
                continue
            variant_name = "custom" if feature == "pronunciation_control" else file_token(variant)
            path = output / f"{feature}_{variant_name}.wav"
            sf.write(path, audio, rate, subtype="PCM_16")
            values.append(
                {
                    "feature": feature,
                    "status": "ok",
                    "variant": variant,
                    "path": str(path),
                    "timing": timing,
                    "resources": resources,
                    "quality": quality,
                }
            )
    if module.CAPABILITIES.get("native_streaming") and not any(
        item["feature"] == "native_streaming" for item in values
    ):
        values.append({"feature": "native_streaming", "status": "ok"})
    return values


def run(args):
    config = benchmark_config()
    set_seed(config["seed"])
    profile = config["profiles"][args.profile]
    set_threads(profile["threads"])
    module = importlib.import_module(MODULES[args.model])
    metadata = models_manifest()["tts"][args.model]
    interval = config["sampling_interval_ms"] / 1000
    baseline = psutil.Process().memory_info().rss
    load_sampler = ResourceSampler(interval).start()
    load_started = time.perf_counter()
    model = module.load(model_dir("tts", args.model), profile["threads"])
    load_s = time.perf_counter() - load_started
    load_resources = load_sampler.stop()
    texts = json.loads((CONFIG / "tts_texts.json").read_text(encoding="utf-8"))
    scope = "cold" if args.cold else args.scope
    if args.smoke:
        scope = args.scope
    wanted = ["short"] if args.smoke else config["suite_scopes"]["tts"][scope]
    texts = [item for item in texts if item["id"] in wanted]
    reference = str(DATA / "tts" / "reference_voice.wav")
    warm_reference = None
    warmups = 0 if args.cold else profile["warmups"]
    repetitions = 1 if args.smoke or args.cold else profile["repetitions"]
    for _ in range(warmups):
        module.generate(model, texts[0]["text"], voice=metadata["voice"], reference=warm_reference)
    output = OUTPUTS / "audio" / args.model
    output.mkdir(parents=True, exist_ok=True)
    measured = []
    for item in texts:
        item_repetitions = repetitions if scope != "full" or item["id"] == "short" else 1
        for repetition in range(item_repetitions):
            audio, rate, timing, resources, quality = render(
                module,
                model,
                item["text"],
                metadata["voice"],
                1.0,
                warm_reference,
                interval,
            )
            suffix = "smoke" if args.smoke else "cold" if args.cold else str(repetition)
            path = output / item["id"] / f"{args.profile}_{suffix}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, audio, rate, subtype="PCM_16")
            measured.append(
                {
                    "repetition": repetition,
                    "test_id": item["id"],
                    "category": item["category"],
                    "text": item["text"],
                    "path": str(path),
                    "timing": timing,
                    "resources": resources,
                    "quality": quality,
                }
            )
    special = (
        []
        if args.cold or scope != "full"
        else special_tests(
            args.model,
            module,
            model,
            reference,
            interval,
            output,
            smoke=args.smoke,
        )
    )
    return {
        "schema_version": 1,
        "status": "ok",
        "kind": "tts",
        "model": args.model,
        "profile": args.profile,
        "scope": scope,
        "phase": "cold" if args.cold else "warm",
        "smoke": args.smoke,
        "environment": environment(),
        "configuration": {
            "threads": profile["threads"],
            "voice": metadata["voice"],
            "scope": scope,
            "generation": module.GENERIC_CONTROLS,
        },
        "baseline_rss_bytes": baseline,
        "load": {
            "seconds": load_s,
            "memory_delta_bytes": load_resources.get("peak_rss_bytes", baseline) - baseline,
            "resources": load_resources,
        },
        "warmups": warmups,
        "repetitions": repetitions,
        "model_disk_bytes": directory_size(model_dir("tts", args.model)),
        "special_capabilities": capability_rows(module, special),
        "special_tests": special,
        "per_test": measured,
        "summary": {
            "first_audio_s": repetition_stats(
                measured, lambda item: item["timing"]["first_audio_s"]
            ),
            "rtf": repetition_stats(measured, lambda item: item["timing"]["rtf"]),
            "peak_rss_bytes": repetition_stats(
                measured, lambda item: item["resources"].get("peak_rss_bytes")
            ),
            "average_rss_bytes": repetition_stats(
                measured, lambda item: item["resources"].get("average_rss_bytes")
            ),
            "average_cpu_percent": repetition_stats(
                measured, lambda item: item["resources"].get("average_cpu_percent")
            ),
            "peak_cpu_percent": repetition_stats(
                measured, lambda item: item["resources"].get("peak_cpu_percent")
            ),
            "cpu_time_s": repetition_stats(
                measured, lambda item: item["resources"].get("cpu_time_s")
            ),
            "clipping_ratio": repetition_stats(
                measured, lambda item: item["quality"]["clipping_ratio"]
            ),
            "silence_ratio": repetition_stats(
                measured, lambda item: item["quality"]["silence_ratio"]
            ),
            "failure_rate": 0.0,
        },
    }


def main():
    suppress_native_dialogs()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODULES, required=True)
    parser.add_argument("--profile", choices=["deployment", "controlled"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scope", choices=["full", "controlled", "cold"], default="full")
    parser.add_argument("--cold", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    try:
        value = run(args)
    except Exception as error:
        value = {
            "schema_version": 1,
            "status": "failed",
            "kind": "tts",
            "model": args.model,
            "profile": args.profile,
            "scope": args.scope,
            "phase": "cold" if args.cold else "warm",
            "smoke": args.smoke,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
    write_json(args.output, value)
    if value["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
