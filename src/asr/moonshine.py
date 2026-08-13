import time
from pathlib import Path

import numpy as np


def find_bundle(path):
    for candidate in Path(path).rglob("tokenizer.bin"):
        return candidate.parent
    raise FileNotFoundError(f"moonshine bundle not found under {path}")


def load(model_path, model_arch, threads):
    from moonshine_voice import ModelArch, Transcriber

    options = {}
    if threads > 0:
        options["intra_op_num_threads"] = threads
    return Transcriber(
        model_path=str(find_bundle(model_path)),
        model_arch=ModelArch(model_arch),
        update_interval=0.5,
        options=options,
    )


def lines_text(transcript):
    return " ".join(
        line.text.strip() for line in getattr(transcript, "lines", []) if line.text.strip()
    )


def offline(model, audio, sample_rate, prompt=None):
    started = time.perf_counter()
    transcript = model.transcribe_without_streaming(audio.astype(np.float32), sample_rate)
    total = time.perf_counter() - started
    return {
        "text": lines_text(transcript),
        "first_partial_s": None,
        "first_final_s": total,
        "endpoint_to_final_s": total,
        "timestamps": [
            {
                "start_s": getattr(line, "start_time", None),
                "end_s": getattr(line, "end_time", None),
                "text": line.text,
            }
            for line in getattr(transcript, "lines", [])
        ],
    }


def streaming(model, audio, sample_rate, chunk_seconds=0.1, prompt=None):
    from moonshine_voice import TranscriptEventListener

    started = time.perf_counter()
    events = []

    class Listener(TranscriptEventListener):
        def on_line_text_changed(self, event):
            events.append(("partial", time.perf_counter() - started, event.line.text))

        def on_line_completed(self, event):
            events.append(("final", time.perf_counter() - started, event.line.text))

    listener = Listener()
    model.add_listener(listener)
    model.start()
    chunk = max(1, round(chunk_seconds * sample_rate))
    for offset in range(0, len(audio), chunk):
        model.add_audio(audio[offset : offset + chunk].astype(np.float32), sample_rate)
    endpoint = time.perf_counter()
    transcript = model.stop()
    finished = time.perf_counter()
    model.remove_listener(listener)
    partials = [event for event in events if event[0] == "partial" and event[2].strip()]
    finals = [event for event in events if event[0] == "final" and event[2].strip()]
    return {
        "text": lines_text(transcript),
        "first_partial_s": partials[0][1] if partials else None,
        "first_final_s": finals[0][1] if finals else finished - started,
        "endpoint_to_final_s": finished - endpoint,
        "events": [{"type": kind, "time_s": stamp, "text": text} for kind, stamp, text in events],
    }
