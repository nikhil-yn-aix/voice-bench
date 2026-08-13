from src.common.config import CONFIG, load_json

RUNTIME_GROUPS = {
    "moonshine_small": "moonshine",
    "moonshine_medium": "moonshine",
    "moonshine_base": "moonshine",
    "whisper_small_en": "whisper",
    "parakeet_110m": "parakeet",
}


def runtime_group(model):
    return RUNTIME_GROUPS.get(model, model)


def model_entries(kind=None):
    manifest = load_json(CONFIG / "models.json")
    kinds = [kind] if kind else ["asr", "tts"]
    return [
        {"kind": current, "id": identifier, **item}
        for current in kinds
        for identifier, item in manifest[current].items()
    ]


def data_entries():
    return load_json(CONFIG / "data.json")["scenarios"]


def validate_registry():
    models = model_entries()
    data = data_entries()
    model_ids = [(item["kind"], item["id"]) for item in models]
    data_ids = [item["id"] for item in data]
    if len(model_ids) != len(set(model_ids)):
        raise ValueError("duplicate model id")
    if len(data_ids) != len(set(data_ids)):
        raise ValueError("duplicate data id")
    for item in models:
        for field in [
            "name",
            "repository",
            "runtime",
            "artifact",
            "license",
            "downloader",
            "capabilities",
        ]:
            if field not in item:
                raise ValueError(f"{item['id']} missing {field}")
    for item in data:
        if "category" not in item:
            raise ValueError(f"{item['id']} missing category")
        if "source" in item and "revision" not in item:
            raise ValueError(f"{item['id']} missing revision")
    return True
