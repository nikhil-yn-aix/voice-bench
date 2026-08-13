import math
import re
import string
from collections import Counter

import jiwer
import numpy as np


def normalize_text(text):
    text = text.lower().replace("\u2019", "'")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return " ".join(text.split())


def error_metrics(reference, hypothesis):
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    words = jiwer.process_words(ref, hyp)
    chars = jiwer.process_characters(ref, hyp)
    return {
        "reference_raw": reference,
        "hypothesis_raw": hypothesis,
        "reference_normalized": ref,
        "hypothesis_normalized": hyp,
        "wer": words.wer,
        "cer": chars.cer,
        "substitutions": words.substitutions,
        "deletions": words.deletions,
        "insertions": words.insertions,
        "hits": words.hits,
        "punctuation_f1": punctuation_f1(reference, hypothesis),
    }


def punctuation_f1(reference, hypothesis):
    ref = Counter(char for char in reference if char in string.punctuation)
    hyp = Counter(char for char in hypothesis if char in string.punctuation)
    true_positive = sum((ref & hyp).values())
    predicted = sum(hyp.values())
    expected = sum(ref.values())
    if predicted == 0 and expected == 0:
        return 1.0
    if predicted == 0 or expected == 0:
        return 0.0
    precision = true_positive / predicted
    recall = true_positive / expected
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def term_recall(reference, hypothesis, terms):
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    present = [normalize_text(term) for term in terms if normalize_text(term) in ref]
    if not present:
        return None
    return sum(term in hyp for term in present) / len(present)


def audio_metrics(audio, sample_rate):
    values = np.asarray(audio, dtype=np.float64).reshape(-1)
    if len(values) == 0:
        return {"sample_rate": sample_rate, "duration_s": 0.0, "empty": True}
    peak = float(np.max(np.abs(values)))
    rms = float(np.sqrt(np.mean(values**2)))
    loudness = 20 * math.log10(max(rms, 1e-12))
    return {
        "sample_rate": sample_rate,
        "duration_s": len(values) / sample_rate,
        "samples": len(values),
        "peak": peak,
        "rms": rms,
        "loudness_dbfs": loudness,
        "clipping_ratio": float(np.mean(np.abs(values) >= 0.999)),
        "silence_ratio": float(np.mean(np.abs(values) < 10 ** (-50 / 20))),
        "dc_offset": float(np.mean(values)),
        "finite": bool(np.isfinite(values).all()),
    }
