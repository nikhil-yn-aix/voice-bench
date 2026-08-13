import csv
import json
import math
import statistics
from datetime import UTC, datetime
from pathlib import Path

from src.common.config import OUTPUTS, benchmark_config, ensure_layout, models_manifest
from src.common.system import environment


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def stats(values):
    clean = [float(value) for value in values if value is not None and math.isfinite(value)]
    if not clean:
        return {}
    ordered = sorted(clean)
    position = math.ceil(0.95 * len(ordered)) - 1
    return {
        "count": len(clean),
        "mean": statistics.fmean(clean),
        "median": statistics.median(clean),
        "p95": ordered[max(0, position)],
        "stddev": statistics.stdev(clean) if len(clean) > 1 else 0.0,
        "min": ordered[0],
        "max": ordered[-1],
    }


def repetition_stats(rows, getter):
    grouped = {}
    for row in rows:
        value = getter(row)
        if value is not None and math.isfinite(value):
            grouped.setdefault(row["test_id"], []).append(value)
    output = stats([statistics.fmean(values) for values in grouped.values() if values])
    repeated = [statistics.stdev(values) for values in grouped.values() if len(values) > 1]
    if output:
        output["stddev"] = statistics.fmean(repeated) if repeated else 0.0
        output["trials"] = sum(len(values) for values in grouped.values())
        output["repeated_scenarios"] = len(repeated)
    return output


def aggregate_runs(runs):
    numeric = {}
    for run in runs:
        flatten_numeric(run, "", numeric)
    return {key: stats(values) for key, values in numeric.items() if values}


def flatten_numeric(value, prefix, output):
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        output.setdefault(prefix, []).append(value)
    elif isinstance(value, dict):
        for key, child in value.items():
            flatten_numeric(child, f"{prefix}.{key}".strip("."), output)


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def collect_raw():
    values = []
    for path in sorted((OUTPUTS / "raw").glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            value["_raw_path"] = str(path)
            values.append(value)
        except json.JSONDecodeError:
            values.append({"status": "invalid_json", "path": str(path)})
    return values


def metric_value(model, key):
    aggregate = model.get("warm_metrics", {})
    candidate = aggregate.get(key, {})
    return candidate.get("mean") if isinstance(candidate, dict) else candidate


def normalized_cost(values):
    finite = [value for value in values.values() if value is not None and math.isfinite(value)]
    if not finite:
        return {key: None for key in values}
    low, high = min(finite), max(finite)
    if high == low:
        return {key: 0.0 if value is not None else None for key, value in values.items()}
    return {
        key: None if value is None else (value - low) / (high - low)
        for key, value in values.items()
    }


def rank(kind, models, profiles, memory_budget_mb=None):
    keys = {
        "asr": {
            "quality": "summary.wer.mean",
            "latency": "summary.first_final_s.mean",
            "memory": "summary.peak_rss_bytes.max",
            "cpu": "summary.cpu_time_s.mean",
            "stability": "summary.total_s.stddev",
            "size": "model_disk_bytes",
        },
        "tts": {
            "quality": "summary.asr_wer.mean",
            "latency": "summary.first_audio_s.mean",
            "memory": "summary.peak_rss_bytes.max",
            "cpu": "summary.cpu_time_s.mean",
            "stability": "summary.rtf.stddev",
            "size": "model_disk_bytes",
        },
    }[kind]
    costs = {
        category: normalized_cost(
            {name: metric_value(model, key) for name, model in models.items()}
        )
        for category, key in keys.items()
    }
    output = {}
    for profile, weights in profiles.items():
        scores = {}
        for name in models:
            required = ["quality", "latency", "memory", "cpu", "size"]
            if any(costs[key][name] is None for key in required):
                scores[name] = None
                continue
            available = [
                (weights[key], costs[key][name]) for key in weights if costs[key][name] is not None
            ]
            if not available:
                scores[name] = None
                continue
            total_weight = sum(weight for weight, _ in available)
            scores[name] = 1 - sum(weight * cost for weight, cost in available) / total_weight
        ordered = sorted(
            scores, key=lambda name: -scores[name] if scores[name] is not None else math.inf
        )
        limit = None if memory_budget_mb is None else memory_budget_mb * 1024**2
        eligible = {
            name: scores[name] is not None
            and (limit is None or metric_value(models[name], keys["memory"]) <= limit)
            for name in models
        }
        output[profile] = {
            "scores": scores,
            "order": ordered,
            "eligible": eligible,
            "eligible_order": [name for name in ordered if eligible[name]],
            "weights": weights,
        }
    return output


def build_master():
    ensure_layout()
    raw = collect_raw()
    manifest = models_manifest()
    config = benchmark_config()
    master = {
        "environment": environment(),
        "benchmark_config": config,
        "asr": {"models": {}},
        "tts": {"models": {}},
        "rankings": {},
        "generated_at": datetime.now(UTC).isoformat(),
    }
    for kind in ["asr", "tts"]:
        for name, metadata in manifest[kind].items():
            runs = [item for item in raw if item.get("kind") == kind and item.get("model") == name]
            benchmark_runs = [item for item in runs if item.get("smoke") is False]
            smoke_runs = [item for item in runs if item.get("smoke") is True]
            successes = [item for item in benchmark_runs if item.get("status") == "ok"]
            warm = [item for item in successes if item.get("phase", "warm") == "warm"]
            cold = [item for item in successes if item.get("phase") == "cold"]
            primary = [
                item
                for item in warm
                if item.get("profile") == "deployment"
                and (kind == "tts" or item.get("mode") == "native")
                and item.get("scope", "full") == "full"
            ]
            failures = [item for item in benchmark_runs if item.get("status") != "ok"]
            configurations = {}
            for item in warm:
                key = "/".join(
                    value
                    for value in [item.get("profile"), item.get("mode"), item.get("scope")]
                    if value
                )
                configurations.setdefault(key, []).append(item)
            master[kind]["models"][name] = {
                "metadata": metadata,
                "configuration": next(
                    (item.get("configuration", {}) for item in reversed(primary)), {}
                ),
                "configuration_metrics": {
                    key: aggregate_runs(items) for key, items in configurations.items()
                },
                "capability_results": next(
                    (item.get("special_capabilities", []) for item in reversed(warm)), []
                ),
                "special_test_results": next(
                    (item.get("special_tests", []) for item in reversed(warm)), []
                ),
                "cold_start_metrics": aggregate_runs(cold),
                "warm_metrics": aggregate_runs(primary),
                "per_test_results": [
                    {
                        **test,
                        "benchmark_profile": item.get("profile"),
                        "benchmark_mode": item.get("mode"),
                        "benchmark_scope": item.get("scope"),
                    }
                    for item in warm
                    for test in item.get("per_test", [])
                ],
                "failures": [
                    {
                        "path": item.get("_raw_path"),
                        "profile": item.get("profile"),
                        "mode": item.get("mode"),
                        "phase": item.get("phase"),
                        "scope": item.get("scope"),
                        "error_type": item.get("error_type"),
                        "error": item.get("error"),
                    }
                    for item in failures
                ],
                "raw_results": [item.get("_raw_path") for item in runs],
                "smoke_results": [
                    {
                        "status": item.get("status"),
                        "path": item.get("_raw_path"),
                        "profile": item.get("profile"),
                        "mode": item.get("mode"),
                        "phase": item.get("phase"),
                        "scope": item.get("scope"),
                        "error": item.get("error"),
                    }
                    for item in smoke_runs
                ],
            }
    quality_path = OUTPUTS / "processed" / "tts_intelligibility.json"
    if quality_path.exists():
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
        for name, rows in quality.get("models", {}).items():
            if name in master["tts"]["models"]:
                master["tts"]["models"][name]["objective_intelligibility"] = rows
                master["tts"]["models"][name]["warm_metrics"]["summary.asr_wer.mean"] = stats(
                    [row["wer"] for row in rows if row.get("status") == "ok"]
                )
    for kind in ["asr", "tts"]:
        master["rankings"][kind] = rank(
            kind,
            master[kind]["models"],
            config["ranking_profiles"],
            config["memory_budget_mb"],
        )
    write_json(OUTPUTS / "master.json", master)
    export_csv(master)
    return master


def export_csv(master):
    summary = []
    capabilities = []
    failures = []
    per_test = []
    for kind in ["asr", "tts"]:
        for name, model in master[kind]["models"].items():
            summary.append(
                {
                    "kind": kind,
                    "model": name,
                    "measured": bool(model["warm_metrics"]),
                    "quality_error": metric_value(model, "summary.wer.mean")
                    if kind == "asr"
                    else metric_value(model, "summary.asr_wer.mean"),
                    "rtf": metric_value(model, "summary.rtf.mean"),
                    "latency_s": metric_value(
                        model,
                        "summary.first_final_s.mean"
                        if kind == "asr"
                        else "summary.first_audio_s.mean",
                    ),
                    "peak_rss_mb": divide(metric_value(model, "summary.peak_rss_bytes.max")),
                    "average_rss_mb": divide(metric_value(model, "summary.average_rss_bytes.mean")),
                    "cpu_time_s": metric_value(model, "summary.cpu_time_s.mean"),
                    "model_disk_mb": divide(metric_value(model, "model_disk_bytes")),
                    "failures": len(model["failures"]),
                }
            )
            for item in model["capability_results"]:
                capabilities.append(
                    {
                        "kind": kind,
                        "model": name,
                        "feature": item["feature"],
                        "supported": item["supported"],
                        "tested": item["tested"],
                        "status": item.get("result", {}).get("status"),
                    }
                )
            for item in model["failures"]:
                failures.append({"kind": kind, "model": name, **item})
            for item in model["per_test_results"]:
                per_test.append(
                    {
                        "kind": kind,
                        "model": name,
                        "test_id": item["test_id"],
                        "profile": item.get("benchmark_profile"),
                        "mode": item.get("benchmark_mode"),
                        "scope": item.get("benchmark_scope"),
                        "repetition": item["repetition"],
                        "wer": item.get("accuracy", {}).get("wer"),
                        "cer": item.get("accuracy", {}).get("cer"),
                        "rtf": item.get("timing", {}).get("rtf"),
                        "peak_rss_mb": divide(item.get("resources", {}).get("peak_rss_bytes")),
                        "cpu_time_s": item.get("resources", {}).get("cpu_time_s"),
                        "path": item.get("path"),
                    }
                )
    rankings = []
    for kind, profiles in master["rankings"].items():
        for profile, result in profiles.items():
            for position, name in enumerate(result["order"], 1):
                rankings.append(
                    {
                        "kind": kind,
                        "profile": profile,
                        "rank": position,
                        "model": name,
                        "score": result["scores"][name],
                        "eligible": result["eligible"][name],
                    }
                )
    write_csv(
        OUTPUTS / "processed" / "summary.csv",
        summary,
        [
            "kind",
            "model",
            "measured",
            "quality_error",
            "rtf",
            "latency_s",
            "peak_rss_mb",
            "average_rss_mb",
            "cpu_time_s",
            "model_disk_mb",
            "failures",
        ],
    )
    write_csv(
        OUTPUTS / "processed" / "rankings.csv",
        rankings,
        ["kind", "profile", "rank", "model", "score", "eligible"],
    )
    write_csv(
        OUTPUTS / "processed" / "capabilities.csv",
        capabilities,
        ["kind", "model", "feature", "supported", "tested", "status"],
    )
    write_csv(
        OUTPUTS / "processed" / "failures.csv",
        failures,
        ["kind", "model", "path", "profile", "mode", "scope", "phase", "error_type", "error"],
    )
    write_csv(
        OUTPUTS / "processed" / "per_test.csv",
        per_test,
        [
            "kind",
            "model",
            "test_id",
            "profile",
            "mode",
            "scope",
            "repetition",
            "wer",
            "cer",
            "rtf",
            "peak_rss_mb",
            "cpu_time_s",
            "path",
        ],
    )


def divide(value):
    return None if value is None else value / 1024**2
