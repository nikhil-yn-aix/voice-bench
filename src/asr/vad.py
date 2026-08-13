import numpy as np

from src.common.config import MODELS


def probabilities(audio):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(
        str(MODELS / "vad" / "silero_vad.onnx"),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    state = np.zeros((2, 1, 128), dtype=np.float32)
    context = np.zeros((1, 64), dtype=np.float32)
    values = []
    padded = np.pad(audio.astype(np.float32), (0, (-len(audio)) % 512))
    for offset in range(0, len(padded), 512):
        chunk = padded[offset : offset + 512][None, :]
        combined = np.concatenate([context, chunk], axis=1)
        output, state = session.run(
            None,
            {"input": combined, "state": state, "sr": np.array(16000, dtype=np.int64)},
        )
        context = combined[:, -64:]
        values.append(float(np.asarray(output).reshape(-1)[0]))
    return values


def segments(audio, config):
    values = probabilities(audio)
    threshold = config["threshold"]
    min_silence = round(config["min_silence_duration_ms"] / 32)
    min_speech = round(config["min_speech_duration_ms"] / 32)
    pad = round(config["speech_pad_ms"] / 32)
    maximum = round(config["max_speech_duration_s"] / 0.032)
    spans = []
    start = None
    silence = 0
    for index, value in enumerate(values):
        if value >= threshold:
            if start is None:
                start = index
            silence = 0
        elif start is not None:
            silence += 1
            if silence >= min_silence or index - start >= maximum:
                end = index - silence + 1
                if end - start >= min_speech:
                    spans.append((max(0, start - pad) * 512, min(len(audio), (end + pad) * 512)))
                start = None
                silence = 0
    if start is not None:
        end = len(values)
        if end - start >= min_speech:
            spans.append((max(0, start - pad) * 512, len(audio)))
    return spans


def sherpa_segments(audio, config):
    import sherpa_onnx

    silero = sherpa_onnx.SileroVadModelConfig(
        model=str(MODELS / "vad" / "silero_vad.onnx"),
        threshold=config["threshold"],
        min_silence_duration=config["min_silence_duration_ms"] / 1000,
        min_speech_duration=config["min_speech_duration_ms"] / 1000,
        max_speech_duration=config["max_speech_duration_s"],
    )
    model = sherpa_onnx.VadModelConfig(
        silero_vad=silero,
        sample_rate=16000,
        num_threads=1,
        provider="cpu",
    )
    detector = sherpa_onnx.VoiceActivityDetector(model, buffer_size_in_seconds=600)
    detector.accept_waveform(audio)
    detector.flush()
    padding = round(config["speech_pad_ms"] * 16)
    spans = []
    while not detector.empty():
        segment = detector.front
        start = max(0, segment.start - padding)
        end = min(len(audio), segment.start + len(segment.samples) + padding)
        if spans and start <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], end))
        else:
            spans.append((start, end))
        detector.pop()
    return spans
