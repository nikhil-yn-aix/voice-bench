import argparse
import json
import traceback

import soundfile as sf
from scipy.signal import resample_poly

from src.asr import whisper
from src.common.config import CONFIG, OUTPUTS, model_dir
from src.common.metrics import error_metrics
from src.common.results import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUTS / "processed" / "tts_intelligibility.json"))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    texts = {
        item["id"]: item["text"]
        for item in json.loads((CONFIG / "tts_texts.json").read_text(encoding="utf-8"))
    }
    model = whisper.load(model_dir("asr", "whisper_small_en"), None, args.threads)
    output = {"evaluator": "Whisper Small.en Q5_1", "models": {}}
    for path in sorted(OUTPUTS.glob("audio/*/*/deployment_0.wav")):
        name = path.parents[1].name
        if name == "blind":
            continue
        test_id = path.parent.name
        try:
            audio, rate = sf.read(path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if rate != 16000:
                audio = resample_poly(audio, 16000, rate).astype("float32")
                rate = 16000
            result = whisper.offline(model, audio, rate)
            metrics = error_metrics(texts[test_id], result["text"])
            row = {
                "status": "ok",
                "test_id": test_id,
                "path": str(path),
                "transcript": result["text"],
                "wer": metrics["wer"],
                "cer": metrics["cer"],
            }
        except Exception as error:
            row = {
                "status": "failed",
                "test_id": test_id,
                "path": str(path),
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        output["models"].setdefault(name, []).append(row)
    write_json(args.output, output)


if __name__ == "__main__":
    main()
