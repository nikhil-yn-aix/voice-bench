import argparse
import bz2
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import urllib.request

from src.common.config import ENVS, MODELS, ensure_layout
from src.common.registry import model_entries, runtime_group, validate_registry
from src.common.results import write_json


def fetch(url, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size:
        return target
    temporary = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "voice-bench/0.1"})
    with urllib.request.urlopen(request, timeout=120) as source, temporary.open("wb") as output:
        while block := source.read(1024 * 1024):
            output.write(block)
    temporary.replace(target)
    return target


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def revisions(target):
    values = set()
    for path in target.rglob("*.metadata"):
        lines = path.read_text(encoding="utf-8").splitlines()
        if lines and len(lines[0]) == 40:
            values.add(lines[0])
    for path in target.rglob("refs/main"):
        value = path.read_text(encoding="utf-8").strip()
        if len(value) == 40:
            values.add(value)
    if not values:
        snapshots = [path.name for path in target.rglob("snapshots/*") if path.is_dir()]
        values.update(value for value in snapshots if len(value) == 40)
    return sorted(values)


def artifact_files(target):
    ignored = {".agent_harnesses.json", ".gitignore", "CACHEDIR.TAG"}
    return [
        path
        for path in target.rglob("*")
        if path.is_file() and path.name not in ignored and path.suffix not in {".log", ".metadata"}
    ]


def verify_artifacts(target, item):
    handler = item["downloader"]
    if handler == "hf_files":
        missing = [name for name in item["files"] if not (target / name).is_file()]
    elif handler == "archive":
        missing = [item["artifact"]] if not (target / item["artifact"]).is_file() else []
        if not list(target.rglob("*.onnx")):
            missing.append("extracted onnx model")
    elif handler == "moonshine":
        missing = [] if list(target.rglob("*.ort")) else ["moonshine ort bundle"]
    else:
        extensions = {".onnx", ".pt", ".pth", ".safetensors"}
        missing = (
            [] if any(path.suffix in extensions for path in target.rglob("*")) else ["weights"]
        )
    if missing:
        raise FileNotFoundError(f"{item['id']} missing {', '.join(missing)}")


def resolve_existing(kind, model, item):
    target = MODELS / kind / model
    verify_artifacts(target, item)
    files = artifact_files(target)
    if not files:
        raise FileNotFoundError(f"no cached artifacts for {model}")
    stored = [path for path in files if not path.is_symlink()]
    found_revisions = revisions(target)
    return {
        "kind": kind,
        "model": model,
        "resolved_revision": found_revisions[0] if len(found_revisions) == 1 else None,
        "resolved_revisions": found_revisions,
        "bytes": sum(path.stat().st_size for path in stored),
        "files": [
            {
                "path": str(path.relative_to(target)),
                "bytes": path.stat().st_size,
                "sha256": digest(path),
                "link": path.is_symlink(),
            }
            for path in sorted(files)
        ],
    }


def recorded_model(kind, model):
    path = MODELS / "resolved.json"
    if not path.is_file():
        return None
    values = json.loads(path.read_text(encoding="utf-8"))
    return next(
        (item for item in values if item.get("kind") == kind and item.get("model") == model),
        None,
    )


def verified_existing(kind, model, item):
    current = resolve_existing(kind, model, item)
    recorded = recorded_model(kind, model)
    if not recorded:
        return current
    expected = [(row["path"], row["bytes"], row["sha256"]) for row in recorded["files"]]
    actual = [(row["path"], row["bytes"], row["sha256"]) for row in current["files"]]
    if expected != actual:
        raise RuntimeError(f"{model} cache differs from models/resolved.json; use --force")
    return current


def extract_bz2_tar(archive, target):
    target.mkdir(parents=True, exist_ok=True)
    with bz2.open(archive, "rb") as source, tarfile.open(fileobj=source) as bundle:
        bundle.extractall(target, filter="data")


def hf_files(item, target):
    from huggingface_hub import HfApi, hf_hub_download

    info = HfApi().model_info(item["repo_id"], revision=item.get("revision", "main"))
    files = item.get("files")
    if files:
        for filename in files:
            hf_hub_download(
                item["repo_id"],
                filename,
                revision=info.sha,
                local_dir=target,
            )
    return info.sha


def child(group, arguments, cache):
    env = os.environ.copy()
    env["UV_CACHE_DIR"] = str(MODELS.parent / ".uv-cache")
    env["HF_HOME"] = str(cache / "hf")
    env["HUGGINGFACE_HUB_CACHE"] = str(cache / "hf" / "hub")
    env["MOONSHINE_VOICE_CACHE"] = str(cache)
    env["UV_PROJECT_ENVIRONMENT"] = str(ENVS / group)
    command = [
        "uv",
        "run",
        "--frozen",
        "--no-default-groups",
        "--group",
        group,
        "python",
        *arguments,
    ]
    subprocess.run(
        command,
        check=True,
        env=env,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )


def install_group(group):
    env = os.environ.copy()
    env["UV_CACHE_DIR"] = str(MODELS.parent / ".uv-cache")
    env["UV_PROJECT_ENVIRONMENT"] = str(ENVS / group)
    subprocess.run(
        ["uv", "sync", "--frozen", "--no-default-groups", "--group", group],
        check=True,
        env=env,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )


def package_download(model, item, target):
    if model.startswith("moonshine_"):
        child(
            "moonshine",
            [
                "-m",
                "moonshine_voice.download",
                "--language",
                "en",
                "--stt",
                "--model-arch",
                str(item["model_arch"]),
            ],
            target,
        )
    elif model == "pocket":
        child("pocket", [__file__, "--internal", "pocket"], target)
    elif model == "chatterbox":
        child("chatterbox", [__file__, "--internal", "chatterbox"], target)
    elif model == "kitten":
        child("kitten", [__file__, "--internal", "kitten"], target)


def internal(model):
    if model == "pocket":
        from pocket_tts import TTSModel

        value = TTSModel.load_model()
        value.get_state_for_audio_prompt("alba")
        value.get_state_for_audio_prompt("marius")
    elif model == "chatterbox":
        from chatterbox.tts_turbo import ChatterboxTurboTTS

        ChatterboxTurboTTS.from_pretrained(device="cpu", nano=True)
    elif model == "kitten":
        from kittentts import KittenTTS

        KittenTTS("KittenML/kitten-tts-mini-0.8")


def download_one(kind, model, item, force=False):
    target = MODELS / kind / model
    resolved = None
    handler = item["downloader"]
    group = runtime_group(model)
    if not force:
        try:
            return verified_existing(kind, model, item)
        except FileNotFoundError:
            pass
    elif target.exists():
        expected_parent = (MODELS / kind).resolve()
        if target.resolve().parent != expected_parent:
            raise RuntimeError(f"unsafe model path: {target}")
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    install_group(group)
    if handler == "archive":
        archive = fetch(item["url"], target / item["artifact"])
        extract_bz2_tar(archive, target)
    elif handler == "hf_files":
        resolved = hf_files(item, target)
    elif handler in {"moonshine", "package"}:
        package_download(model, item, target)
    else:
        raise ValueError(f"unknown downloader {handler}")
    value = resolve_existing(kind, model, item)
    if resolved:
        value["resolved_revision"] = resolved
        value["resolved_revisions"] = [resolved]
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resolve-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--internal", choices=["pocket", "chatterbox", "kitten"])
    args = parser.parse_args()
    if args.internal:
        internal(args.internal)
        return
    from src.common.console import console, progress

    ensure_layout()
    validate_registry()
    selected = [
        (item["kind"], item["id"], item)
        for item in model_entries()
        if not args.model or item["id"] == args.model
    ]
    if not selected:
        parser.error(f"unknown model {args.model}")
    if args.dry_run:
        print(
            json.dumps(
                [
                    {"kind": kind, "model": name, "artifact": item["artifact"]}
                    for kind, name, item in selected
                ],
                indent=2,
            )
        )
        return
    resolved_path = MODELS / "resolved.json"
    previous = {}
    if resolved_path.exists():
        previous = {}
        for item in json.loads(resolved_path.read_text(encoding="utf-8")):
            item.setdefault(
                "resolved_revisions",
                [item["resolved_revision"]] if item.get("resolved_revision") else [],
            )
            previous[(item["kind"], item["model"])] = item
    with progress() as display:
        task = display.add_task("models", total=len(selected))
        for entry in selected:
            display.update(task, description=f"model {entry[1]}")
            value = (
                resolve_existing(entry[0], entry[1], entry[2])
                if args.resolve_only
                else download_one(*entry, force=args.force)
            )
            previous[(value["kind"], value["model"])] = value
            display.advance(task)
    if not args.resolve_only:
        fetch(
            "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx",
            MODELS / "vad" / "silero_vad.onnx",
        )
    write_json(
        resolved_path,
        sorted(previous.values(), key=lambda item: (item["kind"], item["model"])),
    )
    console.print(f"saved {MODELS / 'resolved.json'}")


if __name__ == "__main__":
    main()
