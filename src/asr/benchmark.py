import argparse
import importlib
import json
import time
import traceback

import numpy as np
import psutil
import soundfile as sf

from src.asr.vad import segments as vad_segments
from src.asr.vad import sherpa_segments
from src.common.config import DATA, ROOT, benchmark_config, model_dir, models_manifest
from src.common.metrics import error_metrics, term_recall
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
    "moonshine_small": "src.asr.moonshine_small",
    "moonshine_medium": "src.asr.moonshine_medium",
    "whisper_small_en": "src.asr.whisper",
    "parakeet_110m": "src.asr.parakeet",
    "moonshine_base": "src.asr.candidate_five",
}


def load_audio(path):
    audio, rate = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != 16000:
        from scipy.signal import resample_poly

        audio = resample_poly(audio, 16000, rate).astype(np.float32)
        rate = 16000
    return audio, rate


def pseudo_stream(module, model, audio, rate, seconds):
    started = time.perf_counter()
    first = None
    events = []
    step = round(seconds * rate)
    result = None
    for end in range(step, len(audio) + step, step):
        result = module.offline(model, audio[: min(end, len(audio))], rate)
        elapsed = time.perf_counter() - started
        if result["text"].strip() and first is None:
            first = elapsed
        events.append({"type": "partial", "time_s": elapsed, "text": result["text"]})
        if end >= len(audio):
            break
    result["first_partial_s"] = first
    result["first_final_s"] = time.perf_counter() - started
    result["endpoint_to_final_s"] = result["first_final_s"]
    result["events"] = events
    return result


def infer(module, model, audio, rate, mode, config):
    started = time.perf_counter()
    vad_time = 0.0
    if mode == "silero_vad":
        vad_started = time.perf_counter()
        spans = (
            sherpa_segments(audio, config["vad"])
            if getattr(module, "VAD_BACKEND", None) == "sherpa"
            else vad_segments(audio, config["vad"])
        )
        vad_time = time.perf_counter() - vad_started
        results = [module.offline(model, audio[start:end], rate) for start, end in spans]
        if not results:
            results = [module.offline(model, audio, rate)]
        output = {
            "text": " ".join(result["text"] for result in results).strip(),
            "first_partial_s": None,
            "first_final_s": vad_time + results[0]["first_final_s"],
            "endpoint_to_final_s": results[-1]["endpoint_to_final_s"],
            "vad_segments": [
                {"start_s": start / rate, "end_s": end / rate} for start, end in spans
            ],
        }
    elif mode == "continuous_chunked" and not module.NATIVE_STREAMING:
        output = pseudo_stream(module, model, audio, rate, config["pseudo_stream_seconds"])
    elif module.NATIVE_STREAMING:
        output = module.streaming(model, audio, rate, chunk_seconds=config["stream_chunk_seconds"])
    else:
        output = module.offline(model, audio, rate)
    output["total_s"] = time.perf_counter() - started
    output["vad_time_s"] = vad_time
    output["rtf"] = output["total_s"] / (len(audio) / rate)
    return output


def capabilities(model, mode, module):
    metadata = models_manifest()["asr"][model]
    values = []
    for feature in [
        "native_streaming",
        "partial_transcripts",
        "timestamps",
        "punctuation",
        "capitalization",
        "prompting",
        "silero_vad",
    ]:
        supported = feature in metadata["capabilities"] or feature == "silero_vad"
        tested = supported and (
            feature not in {"native_streaming", "partial_transcripts"} or module.NATIVE_STREAMING
        )
        values.append(
            {
                "feature": feature,
                "supported": supported,
                "tested": tested,
                "result": {"mode": mode} if tested else {},
            }
        )
    return values


def run(args):
    config = benchmark_config()
    set_seed(config["seed"])
    profile = config["profiles"][args.profile]
    set_threads(profile["threads"])
    module = importlib.import_module(MODULES[args.model])
    process = psutil.Process()
    baseline = process.memory_info().rss
    load_sampler = ResourceSampler(config["sampling_interval_ms"] / 1000).start()
    load_started = time.perf_counter()
    model = module.load(
        model_dir("asr", args.model),
        module.MODEL_ARCH,
        profile["threads"],
    )
    load_s = time.perf_counter() - load_started
    load_resources = load_sampler.stop()
    manifest = json.loads((DATA / "asr" / "manifest.json").read_text(encoding="utf-8"))
    scope = "cold" if args.cold else args.scope
    if args.smoke:
        scope = args.scope
    wanted = ["clean"] if args.smoke else config["suite_scopes"]["asr"][scope]
    samples = [item for item in manifest["samples"] if item["id"] in wanted]
    warm_audio, warm_rate = load_audio(ROOT / samples[0]["path"])
    warmups = 0 if args.cold else profile["warmups"]
    repetitions = 1 if args.smoke or args.cold else profile["repetitions"]
    for _ in range(warmups):
        infer(module, model, warm_audio, warm_rate, args.mode, config)
    measured = []
    for sample in samples:
        sample_repetitions = repetitions if scope != "full" or sample["id"] == "clean" else 1
        for repetition in range(sample_repetitions):
            audio, rate = load_audio(ROOT / sample["path"])
            sampler = ResourceSampler(config["sampling_interval_ms"] / 1000).start()
            result = infer(module, model, audio, rate, args.mode, config)
            resources = sampler.stop()
            accuracy = error_metrics(sample["transcript"], result["text"])
            if sample["category"] == "technical_terms":
                terms = [word for word in sample["transcript"].split() if len(word) > 7]
                accuracy["technical_term_recall"] = term_recall(
                    sample["transcript"], result["text"], terms
                )
            measured.append(
                {
                    "repetition": repetition,
                    "test_id": sample["id"],
                    "category": sample["category"],
                    "duration_s": sample["duration_s"],
                    "accuracy": accuracy,
                    "timing": result,
                    "resources": resources,
                }
            )
    return {
        "schema_version": 1,
        "status": "ok",
        "kind": "asr",
        "model": args.model,
        "mode": args.mode,
        "scope": scope,
        "profile": args.profile,
        "phase": "cold" if args.cold else "warm",
        "smoke": args.smoke,
        "environment": environment(),
        "configuration": {
            "threads": profile["threads"],
            "scope": scope,
            "vad": config["vad"] if args.mode == "silero_vad" else None,
        },
        "baseline_rss_bytes": baseline,
        "load": {
            "seconds": load_s,
            "memory_delta_bytes": load_resources.get("peak_rss_bytes", baseline) - baseline,
            "resources": load_resources,
        },
        "warmups": warmups,
        "repetitions": repetitions,
        "model_disk_bytes": directory_size(model_dir("asr", args.model)),
        "special_capabilities": capabilities(args.model, args.mode, module),
        "per_test": measured,
        "summary": {
            "wer": repetition_stats(measured, lambda item: item["accuracy"]["wer"]),
            "cer": repetition_stats(measured, lambda item: item["accuracy"]["cer"]),
            "rtf": repetition_stats(measured, lambda item: item["timing"]["rtf"]),
            "total_s": repetition_stats(measured, lambda item: item["timing"]["total_s"]),
            "first_partial_s": repetition_stats(
                measured, lambda item: item["timing"].get("first_partial_s")
            ),
            "first_final_s": repetition_stats(
                measured, lambda item: item["timing"].get("first_final_s")
            ),
            "endpoint_to_final_s": repetition_stats(
                measured, lambda item: item["timing"].get("endpoint_to_final_s")
            ),
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
        },
    }


def main():
    suppress_native_dialogs()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODULES, required=True)
    parser.add_argument(
        "--mode", choices=["native", "silero_vad", "continuous_chunked"], required=True
    )
    parser.add_argument("--profile", choices=["deployment", "controlled"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--scope", choices=["full", "vad", "stream", "controlled", "cold"], default="full"
    )
    parser.add_argument("--cold", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    try:
        value = run(args)
    except Exception as error:
        value = {
            "schema_version": 1,
            "status": "failed",
            "kind": "asr",
            "model": args.model,
            "mode": args.mode,
            "scope": args.scope,
            "profile": args.profile,
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
