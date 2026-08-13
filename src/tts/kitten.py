import time


def load(model_path, threads):
    from kittentts import KittenTTS

    return KittenTTS(
        "KittenML/kitten-tts-mini-0.8",
        cache_dir=str(model_path / "hf" / "hub"),
    )


def generate(model, text, voice="Jasper", speed=1.0, reference=None, clean_text=True):
    started = time.perf_counter()
    audio = model.generate(text, voice=voice, speed=speed, clean_text=clean_text)
    first = time.perf_counter() - started
    return audio.reshape(-1), 24000, first


CAPABILITIES = {
    "native_streaming": False,
    "voice_cloning": False,
    "multiple_voices": True,
    "speed": True,
    "paralinguistic_tags": False,
    "text_normalization": True,
}

GENERIC_CONTROLS = {"speed": 1.0, "clean_text": True}
