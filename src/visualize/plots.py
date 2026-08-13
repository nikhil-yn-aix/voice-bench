import math
import shutil

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.common.config import OUTPUTS

METRICS = {
    "asr": {
        "wer": ("summary.wer.mean", "word error rate", "asr_wer.png"),
        "cer": ("summary.cer.mean", "character error rate", "asr_cer.png"),
        "rtf": ("summary.rtf.mean", "real-time factor", "asr_rtf.png"),
        "latency": ("summary.total_s.mean", "processing time (s)", "asr_latency.png"),
        "peak_ram": ("summary.peak_rss_bytes.max", "peak rss (mb)", "asr_peak_ram.png"),
        "average_ram": (
            "summary.average_rss_bytes.mean",
            "average rss (mb)",
            "asr_average_ram.png",
        ),
        "cpu": ("summary.cpu_time_s.mean", "cpu time (s)", "asr_cpu_time.png"),
        "cpu_use": (
            "summary.average_cpu_percent.mean",
            "average cpu utilization (%)",
            "asr_average_cpu.png",
        ),
    },
    "tts": {
        "first_audio": (
            "summary.first_audio_s.mean",
            "time to first audio (s)",
            "tts_first_audio.png",
        ),
        "rtf": ("summary.rtf.mean", "real-time factor", "tts_rtf.png"),
        "peak_ram": ("summary.peak_rss_bytes.max", "peak rss (mb)", "tts_peak_ram.png"),
        "average_ram": (
            "summary.average_rss_bytes.mean",
            "average rss (mb)",
            "tts_average_ram.png",
        ),
        "cpu": ("summary.cpu_time_s.mean", "cpu time (s)", "tts_cpu_time.png"),
        "cpu_use": (
            "summary.average_cpu_percent.mean",
            "average cpu utilization (%)",
            "tts_average_cpu.png",
        ),
        "intelligibility": (
            "summary.asr_wer.mean",
            "asr evaluator word error rate",
            "tts_intelligibility_wer.png",
        ),
        "clipping": ("summary.clipping_ratio.mean", "clipping ratio", "tts_clipping.png"),
    },
}

DECISION_KEYS = {
    "asr": {
        "quality": "summary.wer.mean",
        "latency": "summary.first_final_s.mean",
        "memory": "summary.peak_rss_bytes.max",
        "cpu": "summary.cpu_time_s.mean",
        "disk": "model_disk_bytes",
    },
    "tts": {
        "quality": "summary.asr_wer.mean",
        "latency": "summary.first_audio_s.mean",
        "memory": "summary.peak_rss_bytes.max",
        "cpu": "summary.cpu_time_s.mean",
        "disk": "model_disk_bytes",
    },
}


def finite(number):
    return number is not None and math.isfinite(number)


def value(model, key):
    item = model.get("warm_metrics", {}).get(key, {})
    return item.get("mean") if isinstance(item, dict) else item


def error(model, key):
    error_key = key.removesuffix("mean") + "stddev" if key.endswith(".mean") else key
    item = model.get("warm_metrics", {}).get(error_key, {})
    return item.get("mean", 0) if isinstance(item, dict) else 0


def save(figure, filename):
    figure.tight_layout()
    figure.savefig(OUTPUTS / "plots" / filename, dpi=180)
    plt.close(figure)


def chart(models, key, label, filename):
    rows = [(name, value(model, key), error(model, key)) for name, model in models.items()]
    rows = [(name, number, spread) for name, number, spread in rows if finite(number)]
    if not rows:
        return
    names, values, errors = zip(*rows, strict=True)
    if "rss_bytes" in key:
        values = tuple(number / 1024**2 for number in values)
        errors = tuple(number / 1024**2 for number in errors)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(names, values, yerr=errors, capsize=3)
    axis.set_ylabel(label)
    axis.set_title(label)
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.25)
    save(figure, filename)


def score_chart(kind, ranking):
    scores = ranking.get("balanced", {}).get("scores", {})
    rows = [(name, score) for name, score in scores.items() if finite(score)]
    if not rows:
        return
    names, values = zip(*rows, strict=True)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(names, values)
    axis.set_ylim(0, 1)
    axis.set_ylabel("balanced deployment score")
    axis.set_title(f"{kind} deployment comparison")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.25)
    save(figure, f"{kind}_deployment_score.png")


def scatter(models, x_key, y_key, x_label, y_label, title, filename, pareto=False):
    rows = [(name, value(model, x_key), value(model, y_key)) for name, model in models.items()]
    rows = [(name, x, y) for name, x, y in rows if finite(x) and finite(y)]
    if len(rows) < 2:
        return
    names, xs, ys = zip(*rows, strict=True)
    if "bytes" in x_key:
        xs = tuple(number / 1024**2 for number in xs)
    if "bytes" in y_key:
        ys = tuple(number / 1024**2 for number in ys)
    figure, axis = plt.subplots(figsize=(8, 4.8))
    axis.scatter(xs, ys, s=55)
    if pareto:
        frontier = []
        best = math.inf
        for x, y in sorted(zip(xs, ys, strict=True)):
            if y < best:
                frontier.append((x, y))
                best = y
        if len(frontier) > 1:
            axis.plot(*zip(*frontier, strict=True), linewidth=1.5)
    for name, x, y in zip(names, xs, ys, strict=True):
        axis.annotate(name, (x, y), xytext=(5, 4), textcoords="offset points", fontsize=8)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_title(title)
    axis.grid(alpha=0.25)
    save(figure, filename)


def efficiency_chart(kind, models, denominator, label, filename):
    quality_key = DECISION_KEYS[kind]["quality"]
    rows = []
    for name, model in models.items():
        quality = value(model, quality_key)
        cost = value(model, denominator)
        if finite(quality) and finite(cost) and cost > 0:
            scale = cost / 1024**2 if "bytes" in denominator else cost
            rows.append((name, (1 / (1 + max(0.0, quality))) / scale))
    if not rows:
        return
    names, values = zip(*rows, strict=True)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(names, values)
    axis.set_ylabel(label)
    axis.set_title(label)
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.25)
    save(figure, filename)


def scenario_heatmap(kind, models):
    entries = {}
    scenarios = set()
    for name, model in models.items():
        grouped = {}
        for row in model.get("per_test_results", []):
            if row.get("benchmark_profile") != "deployment":
                continue
            if kind == "asr" and row.get("benchmark_mode") != "native":
                continue
            scenario = row.get("category") or row.get("test_id")
            metric = (
                row.get("accuracy", {}).get("wer")
                if kind == "asr"
                else row.get("timing", {}).get("rtf")
            )
            if scenario and finite(metric):
                grouped.setdefault(scenario, []).append(metric)
        entries[name] = {key: sum(values) / len(values) for key, values in grouped.items()}
        scenarios.update(grouped)
    names = [name for name in models if entries.get(name)]
    scenarios = sorted(scenarios)
    if not names or not scenarios:
        return
    matrix = np.array(
        [[entries[name].get(scenario, np.nan) for scenario in scenarios] for name in names]
    )
    figure, axis = plt.subplots(figsize=(max(8, len(scenarios) * 1.2), max(4, len(names) * 0.65)))
    image = axis.imshow(np.ma.masked_invalid(matrix), aspect="auto")
    axis.set_xticks(range(len(scenarios)), scenarios, rotation=30, ha="right")
    axis.set_yticks(range(len(names)), names)
    label = "word error rate" if kind == "asr" else "real-time factor"
    axis.set_title(f"{kind} performance by scenario")
    figure.colorbar(image, ax=axis, label=label)
    save(figure, f"{kind}_scenario_heatmap.png")


def ranking_heatmap(kind, ranking):
    profiles = list(ranking)
    names = sorted({name for profile in ranking.values() for name in profile.get("scores", {})})
    names = [
        name
        for name in names
        if any(finite(ranking[profile].get("scores", {}).get(name)) for profile in profiles)
    ]
    if not profiles or len(names) < 2:
        return
    matrix = np.array(
        [
            [ranking[profile].get("scores", {}).get(name, np.nan) for name in names]
            for profile in profiles
        ],
        dtype=float,
    )
    if not np.isfinite(matrix).any():
        return
    figure, axis = plt.subplots(figsize=(max(8, len(names) * 1.2), max(3.5, len(profiles) * 0.7)))
    image = axis.imshow(np.ma.masked_invalid(matrix), aspect="auto", vmin=0, vmax=1)
    axis.set_xticks(range(len(names)), names, rotation=25, ha="right")
    axis.set_yticks(range(len(profiles)), [name.replace("_", " ") for name in profiles])
    axis.set_title(f"{kind} deployment scores by priority")
    figure.colorbar(image, ax=axis, label="score")
    save(figure, f"{kind}_ranking_heatmap.png")


def tradeoff_paths(kind, models):
    keys = DECISION_KEYS[kind]
    names = [
        name
        for name, model in models.items()
        if all(finite(value(model, key)) for key in keys.values())
    ]
    if len(names) < 2:
        return
    raw = {
        category: {name: value(models[name], key) for name in names}
        for category, key in keys.items()
    }
    normalized = {}
    for category, values in raw.items():
        available = [number for number in values.values() if finite(number)]
        if not available:
            continue
        low, high = min(available), max(available)
        normalized[category] = {
            name: None
            if not finite(number)
            else 1.0
            if high == low
            else 1 - (number - low) / (high - low)
            for name, number in values.items()
        }
    categories = list(normalized)
    if len(categories) < 3:
        return
    figure, axis = plt.subplots(figsize=(8, 4.8))
    x = range(len(categories))
    drawn = False
    for name in names:
        values = [normalized[category][name] for category in categories]
        if all(finite(number) for number in values):
            axis.plot(x, values, marker="o", label=name)
            drawn = True
    if not drawn:
        plt.close(figure)
        return
    axis.set_xticks(list(x), categories)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("relative deployment score")
    axis.set_title(f"{kind} normalized trade-offs")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    save(figure, f"{kind}_tradeoff_paths.png")


def decision_plots(kind, models, ranking):
    quality = DECISION_KEYS[kind]["quality"]
    speed = "summary.rtf.mean"
    scatter(
        models,
        "summary.peak_rss_bytes.max",
        quality,
        "peak rss (mb)",
        "error rate",
        f"{kind} quality and memory trade-off",
        f"{kind}_quality_memory_pareto.png",
        True,
    )
    scatter(
        models,
        speed,
        quality,
        "real-time factor",
        "error rate",
        f"{kind} quality and speed trade-off",
        f"{kind}_quality_speed_pareto.png",
        True,
    )
    scatter(
        models,
        "model_disk_bytes",
        "summary.peak_rss_bytes.max",
        "downloaded model size (mb)",
        "peak rss (mb)",
        f"{kind} disk size and runtime memory",
        f"{kind}_disk_vs_peak_ram.png",
    )
    efficiency_chart(
        kind,
        models,
        "model_disk_bytes",
        "quality per model mb",
        f"{kind}_quality_per_mb.png",
    )
    efficiency_chart(
        kind,
        models,
        "summary.cpu_time_s.mean",
        "quality per cpu second",
        f"{kind}_quality_per_cpu_second.png",
    )
    scenario_heatmap(kind, models)
    ranking_heatmap(kind, ranking)
    tradeoff_paths(kind, models)


def generate_plots(master):
    target = (OUTPUTS / "plots").resolve()
    if target.parent != OUTPUTS.resolve():
        raise ValueError("invalid plot directory")
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for kind, metrics in METRICS.items():
        models = master.get(kind, {}).get("models", {})
        ranking = master.get("rankings", {}).get(kind, {})
        for key, label, filename in metrics.values():
            chart(models, key, label, filename)
        score_chart(kind, ranking)
        decision_plots(kind, models, ranking)
