import time
from pathlib import Path


def locate(path, name):
    matches = list(Path(path).rglob(name))
    if not matches:
        raise FileNotFoundError(f"{name} not found under {path}")
    return matches[0]


def load(model_path, model_arch, threads):
    import sherpa_onnx

    return sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
        model=str(locate(model_path, "model.int8.onnx")),
        tokens=str(locate(model_path, "tokens.txt")),
        num_threads=threads or 4,
        provider="cpu",
        debug=False,
    )


def offline(model, audio, sample_rate, prompt=None):
    stream = model.create_stream()
    stream.accept_waveform(sample_rate, audio)
    started = time.perf_counter()
    model.decode_stream(stream)
    total = time.perf_counter() - started
    result = stream.result
    return {
        "text": result.text,
        "first_partial_s": None,
        "first_final_s": total,
        "endpoint_to_final_s": total,
        "timestamps": getattr(result, "timestamps", []),
    }


streaming = None

MODEL_ARCH = None
NATIVE_STREAMING = False
VAD_BACKEND = "sherpa"
