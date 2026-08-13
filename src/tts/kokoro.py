import time
from pathlib import Path

import numpy as np


def load(model_path, threads):
    from kokoro import KModel, KPipeline

    root = Path(model_path)
    model = KModel(
        repo_id="hexgrad/Kokoro-82M",
        config=str(root / "config.json"),
        model=str(root / "kokoro-v1_0.pth"),
    ).eval()
    pipeline = KPipeline(
        lang_code="a",
        repo_id="hexgrad/Kokoro-82M",
        model=model,
        device="cpu",
    )
    return {
        "pipeline": pipeline,
        "voices": {
            "af_heart": str(root / "voices/af_heart.pt"),
            "am_michael": str(root / "voices/am_michael.pt"),
        },
    }


def generate(state, text, voice="af_heart", speed=1.0, reference=None, phonemes=None):
    started = time.perf_counter()
    chunks = []
    first = None
    selected = state["voices"].get(voice, voice)
    source = (
        state["pipeline"].generate_from_tokens(phonemes, voice=selected, speed=speed)
        if phonemes
        else state["pipeline"](text, voice=selected, speed=speed)
    )
    for result in source:
        audio = result.audio if hasattr(result, "audio") else result[2]
        if first is None:
            first = time.perf_counter() - started
        chunks.append(np.asarray(audio, dtype=np.float32).reshape(-1))
    return np.concatenate(chunks), 24000, first


CAPABILITIES = {
    "native_streaming": False,
    "voice_cloning": False,
    "multiple_voices": True,
    "speed": True,
    "paralinguistic_tags": False,
    "pronunciation_control": True,
}

GENERIC_CONTROLS = {"language": "american_english", "speed": 1.0}
