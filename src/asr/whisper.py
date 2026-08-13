import os
import time
from pathlib import Path

import soundfile as sf

from src.common.config import OUTPUTS


def load(model_path, model_arch, threads):
    from pywhispercpp.model import Model

    path = Path(model_path) / "ggml-small.en-q5_1.bin"
    options = {
        "language": "en",
        "print_realtime": False,
        "print_progress": False,
        "print_timestamps": False,
    }
    if threads > 0:
        options["n_threads"] = threads
    return Model(str(path), context_params={"use_gpu": False}, **options)


def offline(model, audio, sample_rate, prompt=None):
    temporary = OUTPUTS / "raw" / f"whisper_{os.getpid()}_{time.time_ns()}.wav"
    try:
        sf.write(temporary, audio, sample_rate, subtype="PCM_16")
        started = time.perf_counter()
        options = {} if not prompt else {"initial_prompt": prompt}
        segments = model.transcribe(str(temporary), **options)
        total = time.perf_counter() - started
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "text": " ".join(segment.text.strip() for segment in segments),
        "first_partial_s": None,
        "first_final_s": total,
        "endpoint_to_final_s": total,
        "timestamps": [
            {"start_s": segment.t0 / 100, "end_s": segment.t1 / 100, "text": segment.text}
            for segment in segments
        ],
    }


streaming = None

MODEL_ARCH = None
NATIVE_STREAMING = False
