import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import time
from datetime import UTC, datetime

from src.common.config import (
    CONFIG,
    ENVS,
    MODELS,
    OUTPUTS,
    ROOT,
    benchmark_config,
    ensure_layout,
    models_manifest,
)
from src.common.console import console, progress
from src.common.registry import runtime_group
from src.common.results import build_master, write_json
from src.common.review import build_review
from src.visualize.plots import generate_plots


def runtime_env(group, cache):
    env = os.environ.copy()
    env["UV_CACHE_DIR"] = str(ROOT / ".uv-cache")
    env["UV_PROJECT_ENVIRONMENT"] = str(ENVS / group)
    env["HF_HOME"] = str(cache)
    env["HUGGINGFACE_HUB_CACHE"] = str(cache / "hub")
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_DATASETS_OFFLINE"] = "1"
    return env


def raw_name(kind, model, profile, mode, scope, phase):
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    parts = [kind, model, profile]
    if mode:
        parts.append(mode)
    parts.append(scope)
    parts.extend([phase, stamp])
    return OUTPUTS / "raw" / ("_".join(parts) + ".json")


def job_key(kind, model, profile, mode, scope, cold):
    return kind, model, profile, mode, "cold" if cold else scope, "cold" if cold else "warm"


def result_key(value):
    return (
        value.get("kind"),
        value.get("model"),
        value.get("profile"),
        value.get("mode"),
        value.get("scope"),
        value.get("phase", "warm"),
    )


def resume_jobs(jobs):
    raw = OUTPUTS / "raw"
    superseded = raw / "superseded"
    successful = set()
    failed = []
    for path in sorted(raw.glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            failed.append(path)
            continue
        if value.get("smoke") is False and value.get("status") == "ok":
            successful.add(result_key(value))
        elif value.get("smoke") is False:
            failed.append(path)
    if failed:
        superseded.mkdir(parents=True, exist_ok=True)
        for path in failed:
            path.replace(superseded / path.name)
    return [job for job in jobs if job_key(*job) not in successful]


def execute(kind, model, profile, mode, scope, cold, smoke):
    phase = "cold" if cold else "warm"
    output = raw_name(kind, model, profile, mode, scope, phase)
    module = f"src.{kind}.benchmark"
    command = [
        "uv",
        "run",
        "--frozen",
        "--offline",
        "--no-default-groups",
        "--group",
        runtime_group(model),
        "python",
        "-m",
        module,
        "--model",
        model,
        "--profile",
        profile,
        "--output",
        str(output),
        "--scope",
        "cold" if cold else scope,
    ]
    if mode:
        command.extend(["--mode", mode])
    if cold:
        command.append("--cold")
    if smoke:
        command.append("--smoke")
    started = time.perf_counter()
    env = runtime_env(runtime_group(model), MODELS / kind / model / "hf")
    env["MOONSHINE_VOICE_CACHE"] = str(MODELS / kind / model)
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    completed = subprocess.run(command, check=False, env=env, creationflags=creationflags)
    startup = time.perf_counter() - started
    if not output.exists():
        write_json(
            output,
            {
                "schema_version": 1,
                "status": "failed",
                "kind": kind,
                "model": model,
                "profile": profile,
                "mode": mode,
                "phase": phase,
                "scope": scope,
                "smoke": smoke,
                "error": f"worker exited {completed.returncode} without a result",
            },
        )
    value = json.loads(output.read_text(encoding="utf-8"))
    value["process_wall_s"] = startup
    write_json(output, value)
    return completed.returncode == 0


def blind_package():
    blind = OUTPUTS / "audio" / "blind"
    if blind.exists():
        shutil.rmtree(blind)
    blind.mkdir(parents=True)
    texts = {
        item["id"]: item["text"]
        for item in json.loads((CONFIG / "tts_texts.json").read_text(encoding="utf-8"))
    }
    files = sorted(OUTPUTS.glob("audio/*/*/deployment_0.wav"))
    seed = benchmark_config()["seed"]
    random.Random(seed).shuffle(files)
    public = []
    private = []
    for index, source in enumerate(files, 1):
        blind_id = "v" + hashlib.sha256(f"{seed}:{index}".encode()).hexdigest()[:8]
        target = blind / f"{blind_id}.wav"
        shutil.copy2(source, target)
        test_id = source.parent.name
        public.append(
            {"blind_id": blind_id, "file": target.name, "test_id": test_id, "text": texts[test_id]}
        )
        private.append(
            {"blind_id": blind_id, "model": source.parents[1].name, "source": str(source)}
        )
    write_json(blind / "evaluation.json", public)
    write_json(OUTPUTS / "processed" / "blind_key.json", private)


def clear_outputs():
    root = OUTPUTS.resolve()
    for name in ["raw", "processed", "audio", "plots", "review"]:
        target = (OUTPUTS / name).resolve()
        if target.parent != root:
            raise RuntimeError(f"unsafe output path: {target}")
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
    report = OUTPUTS / "report.md"
    if report.exists():
        report.unlink()
    master = OUTPUTS / "master.json"
    if master.exists():
        master.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["asr", "tts", "all"], default="all")
    parser.add_argument("--model")
    parser.add_argument("--profile", choices=["deployment", "controlled", "all"], default="all")
    parser.add_argument(
        "--mode", choices=["native", "silero_vad", "continuous_chunked", "all"], default="all"
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--plots-only", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.fresh and args.resume:
        parser.error("--fresh and --resume cannot be used together")
    if args.resume and (
        args.kind != "all"
        or args.model
        or args.profile != "all"
        or args.mode != "all"
        or args.smoke
        or args.plots_only
    ):
        parser.error("--resume continues the complete measured suite and takes no filters")
    ensure_layout()
    if args.fresh and not args.dry_run:
        clear_outputs()
    if args.plots_only:
        master = build_master()
        generate_plots(master)
        build_review(master)
        return
    manifest = models_manifest()
    kinds = [args.kind] if args.kind != "all" else ["asr", "tts"]
    profiles = [args.profile] if args.profile != "all" else ["deployment", "controlled"]
    modes = [args.mode] if args.mode != "all" else ["native", "silero_vad", "continuous_chunked"]
    jobs = []
    for kind in kinds:
        for model in manifest[kind]:
            if args.model and model != args.model:
                continue
            if args.smoke:
                smoke_profile = args.profile if args.profile != "all" else "deployment"
                smoke_mode = None
                smoke_scope = "full"
                if kind == "asr":
                    smoke_mode = args.mode if args.mode != "all" else "native"
                    smoke_scope = {
                        "native": "full",
                        "silero_vad": "vad",
                        "continuous_chunked": "stream",
                    }[smoke_mode]
                jobs.append((kind, model, smoke_profile, smoke_mode, smoke_scope, False))
            else:
                for profile in profiles:
                    if kind == "tts":
                        scope = "full" if profile == "deployment" else "controlled"
                        jobs.append((kind, model, profile, None, scope, False))
                        continue
                    selected_modes = modes
                    if args.mode == "all" and profile == "controlled":
                        if manifest[kind][model].get("controlled_threads") is False:
                            continue
                        selected_modes = ["native"]
                    for mode in selected_modes:
                        if (
                            args.mode == "all"
                            and mode == "continuous_chunked"
                            and manifest[kind][model]["streaming"]
                        ):
                            continue
                        scope = {
                            "native": "full",
                            "silero_vad": "vad",
                            "continuous_chunked": "stream",
                        }[mode]
                        if profile == "controlled":
                            scope = "controlled"
                        jobs.append((kind, model, profile, mode, scope, False))
            jobs.append(
                (
                    kind,
                    model,
                    "deployment",
                    "native" if kind == "asr" else None,
                    "cold",
                    True,
                )
            )
    random.Random(benchmark_config()["seed"]).shuffle(jobs)
    if args.resume and not args.dry_run:
        jobs = resume_jobs(jobs)
    if args.dry_run:
        console.print_json(
            data={
                "jobs": [
                    {
                        "kind": kind,
                        "model": model,
                        "profile": profile,
                        "mode": mode,
                        "scope": scope,
                        "phase": "cold" if cold else "warm",
                    }
                    for kind, model, profile, mode, scope, cold in jobs
                ],
                "count": len(jobs),
            }
        )
        return
    outcomes = []
    with progress() as display:
        task = display.add_task("benchmarks", total=len(jobs))
        for job in jobs:
            kind, model, profile, mode, scope, cold = job
            phase = "cold" if cold else "warm"
            label = " ".join(value for value in [kind, model, profile, mode, scope, phase] if value)
            display.update(task, description=label)
            outcomes.append(execute(*job, args.smoke))
            display.advance(task)
    if any(job[0] == "tts" for job in jobs):
        blind_package()
        if not args.smoke:
            env = runtime_env("whisper", MODELS / "asr" / "whisper_small_en" / "hf")
            subprocess.run(
                [
                    "uv",
                    "run",
                    "--frozen",
                    "--offline",
                    "--no-default-groups",
                    "--group",
                    "whisper",
                    "python",
                    "-m",
                    "src.tts.evaluate",
                ],
                check=False,
                env=env,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
    master = build_master()
    generate_plots(master)
    build_review(master)
    console.print(f"saved {OUTPUTS / 'processed' / 'master.json'}")
    console.print(f"saved {OUTPUTS / 'plots'}")
    console.print(f"saved {OUTPUTS / 'review'}")
    if not all(outcomes):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
