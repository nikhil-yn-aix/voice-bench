import time

import numpy as np


def load(model_path, threads):
    from pocket_tts import TTSModel

    model = TTSModel.load_model()
    return {"model": model, "voices": {"alba": model.get_state_for_audio_prompt("alba")}}


def generate(state, text, voice="alba", speed=1.0, reference=None):
    model = state["model"]
    key = reference or voice
    if key not in state["voices"]:
        state["voices"][key] = model.get_state_for_audio_prompt(key)
    started = time.perf_counter()
    chunks = []
    first = None
    for chunk in model.generate_audio_stream(state["voices"][key], text):
        if first is None:
            first = time.perf_counter() - started
        chunks.append(chunk.detach().cpu().numpy().reshape(-1))
    return np.concatenate(chunks), model.sample_rate, first


CAPABILITIES = {
    "native_streaming": True,
    "voice_cloning": True,
    "multiple_voices": True,
    "speed": False,
    "paralinguistic_tags": False,
}

GENERIC_CONTROLS = {
    "language": "english",
    "temperature": 0.7,
    "streaming": True,
}
