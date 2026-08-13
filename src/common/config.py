import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config"
MODELS = ROOT / "models"
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
ENVS = ROOT / ".venvs"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def models_manifest():
    return load_json(CONFIG / "models.json")


def benchmark_config():
    return load_json(CONFIG / "benchmark.json")


def ensure_layout():
    paths = [
        MODELS,
        DATA / "asr",
        DATA / "tts",
        OUTPUTS / "raw",
        OUTPUTS / "processed",
        OUTPUTS / "audio",
        OUTPUTS / "plots",
        OUTPUTS / "review",
    ]
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def model_dir(kind, model):
    return MODELS / kind / model
