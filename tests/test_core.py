import ast
import importlib
import io
import os
import tempfile
import tokenize
import unittest
from pathlib import Path

from scripts.run_all import runtime_env
from src.common.config import ENVS, ROOT
from src.common.metrics import audio_metrics, error_metrics, normalize_text, punctuation_f1
from src.common.registry import model_entries, runtime_group, validate_registry
from src.common.results import rank, repetition_stats, stats, write_json
from src.visualize.plots import error as plot_error


class CoreTests(unittest.TestCase):
    def test_source_has_no_comments_or_docstrings(self):
        for root in [ROOT / "scripts", ROOT / "src", ROOT / "tests"]:
            for path in root.rglob("*.py"):
                source = path.read_text(encoding="utf-8")
                tokens = tokenize.generate_tokens(io.StringIO(source).readline)
                self.assertFalse(any(token.type == tokenize.COMMENT for token in tokens), path)
                tree = ast.parse(source)
                nodes = [tree, *ast.walk(tree)]
                self.assertFalse(
                    any(
                        isinstance(
                            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
                        )
                        and ast.get_docstring(node, clean=False) is not None
                        for node in nodes
                    ),
                    path,
                )

    def test_adapters(self):
        names = {
            "asr": [
                "moonshine_small",
                "moonshine_medium",
                "whisper",
                "parakeet",
                "candidate_five",
            ],
            "tts": ["pocket", "kokoro", "chatterbox", "kitten", "piper"],
        }
        for kind, adapters in names.items():
            for adapter in adapters:
                module = importlib.import_module(f"src.{kind}.{adapter}")
                self.assertTrue(callable(module.load))

    def test_registry(self):
        self.assertTrue(validate_registry())
        models = model_entries()
        self.assertEqual(len([item for item in models if item["kind"] == "asr"]), 5)
        self.assertEqual(len([item for item in models if item["kind"] == "tts"]), 5)
        self.assertEqual(runtime_group("whisper_small_en"), "whisper")
        self.assertEqual(runtime_group("moonshine_base"), "moonshine")
        self.assertEqual(runtime_group("piper"), "piper")

    def test_text_metrics(self):
        self.assertEqual(normalize_text("Hello, WORLD!"), "hello world")
        value = error_metrics("one two three", "one four three")
        self.assertAlmostEqual(value["wer"], 1 / 3)
        self.assertEqual(value["substitutions"], 1)
        self.assertEqual(punctuation_f1("Hello.", "Hello?"), 0.0)

    def test_audio_metrics(self):
        value = audio_metrics([0.0, 0.5, -0.5], 3)
        self.assertEqual(value["duration_s"], 1.0)
        self.assertEqual(value["clipping_ratio"], 0.0)

    def test_stats(self):
        value = stats([1, 2, 3])
        self.assertEqual(value["median"], 2)
        self.assertEqual(value["p95"], 3)

    def test_repetition_stats_separates_trials_from_scenarios(self):
        rows = [
            {"test_id": "a", "repetition": 0, "value": 1.0},
            {"test_id": "a", "repetition": 1, "value": 3.0},
            {"test_id": "b", "repetition": 0, "value": 2.0},
            {"test_id": "b", "repetition": 1, "value": 4.0},
        ]
        value = repetition_stats(rows, lambda row: row["value"])
        self.assertEqual(value["mean"], 2.5)
        self.assertAlmostEqual(value["stddev"], 2**0.5)

    def test_ranking_rejects_incomplete_measurements(self):
        complete = {
            "warm_metrics": {
                "summary.wer.mean": {"mean": 0.1},
                "summary.first_final_s.mean": {"mean": 0.2},
                "summary.peak_rss_bytes.max": {"mean": 100},
                "summary.cpu_time_s.mean": {"mean": 0.3},
                "summary.total_s.stddev": {"mean": 0.01},
                "model_disk_bytes": {"mean": 50},
            }
        }
        incomplete = {"warm_metrics": {"model_disk_bytes": {"mean": 10}}}
        profiles = {
            "balanced": {
                "quality": 1,
                "latency": 1,
                "memory": 1,
                "cpu": 1,
                "stability": 1,
                "size": 1,
            }
        }
        result = rank("asr", {"complete": complete, "incomplete": incomplete}, profiles)
        self.assertIsNotNone(result["balanced"]["scores"]["complete"])
        self.assertIsNone(result["balanced"]["scores"]["incomplete"])

    def test_plot_error_uses_measured_run_variability(self):
        model = {"warm_metrics": {"summary.rtf.stddev": {"mean": 0.25, "stddev": 0.0}}}
        self.assertEqual(plot_error(model, "summary.rtf.mean"), 0.25)

    def test_atomic_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value.json"
            write_json(path, {"ok": True})
            self.assertEqual(path.read_text(encoding="utf-8"), '{\n  "ok": true\n}')

    def test_runtime_environment_is_offline_and_isolated(self):
        env = runtime_env("parakeet", ROOT / "models" / "asr" / "parakeet_110m" / "hf")
        self.assertEqual(env["UV_PROJECT_ENVIRONMENT"], str(ENVS / "parakeet"))
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        self.assertEqual(env["TRANSFORMERS_OFFLINE"], "1")
        self.assertEqual(env["HF_DATASETS_OFFLINE"], "1")
        self.assertNotEqual(env["UV_PROJECT_ENVIRONMENT"], os.environ.get("VIRTUAL_ENV"))


if __name__ == "__main__":
    unittest.main()
