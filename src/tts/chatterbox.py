import time


def load(model_path, threads):
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    return ChatterboxTurboTTS.from_pretrained(device="cpu", nano=True)


def generate(model, text, voice="default", speed=1.0, reference=None):
    started = time.perf_counter()
    audio = model.generate(text, audio_prompt_path=reference)
    first = time.perf_counter() - started
    return audio.detach().cpu().numpy().reshape(-1), model.sr, first


CAPABILITIES = {
    "native_streaming": False,
    "voice_cloning": True,
    "multiple_voices": False,
    "speed": False,
    "paralinguistic_tags": True,
}

GENERIC_CONTROLS = {
    "repetition_penalty": 1.2,
    "top_p": 0.95,
    "temperature": 0.8,
    "top_k": 1000,
    "builtin_conditioning": True,
}
