import time
from pathlib import Path

import numpy as np


def load(model_path, threads):
    from piper import PiperVoice

    path = next(Path(model_path).rglob("en_US-lessac-high.onnx"))
    return PiperVoice.load(str(path), use_cuda=False)


def generate(
    model,
    text,
    voice="en_US-lessac-high",
    speed=1.0,
    reference=None,
    noise_scale=None,
    noise_w_scale=None,
):
    from piper.config import SynthesisConfig

    config = SynthesisConfig(
        length_scale=1 / speed,
        noise_scale=noise_scale,
        noise_w_scale=noise_w_scale,
    )
    started = time.perf_counter()
    chunks = []
    first = None
    rate = None
    for chunk in model.synthesize(text, syn_config=config):
        if first is None:
            first = time.perf_counter() - started
        rate = chunk.sample_rate
        chunks.append(
            np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16).astype(np.float32) / 32768
        )
    return np.concatenate(chunks), rate, first


CAPABILITIES = {
    "native_streaming": True,
    "voice_cloning": False,
    "multiple_voices": False,
    "speed": True,
    "paralinguistic_tags": False,
    "noise_control": True,
    "phoneme_width": True,
}

GENERIC_CONTROLS = {
    "speed": 1.0,
    "noise_scale": "voice_config",
    "noise_w_scale": "voice_config",
    "normalize_audio": True,
}
