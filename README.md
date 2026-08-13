# voice bench

cpu-only benchmarks for five asr and five tts models.

- [results and decision](conclusion.md)
- [local dashboard](outputs/review/index.html)
- [master result](outputs/master.json)

## inspect the published run

clone the repository and open `outputs/review/index.html` in a browser. no setup is required. the page includes metrics, plots, asr transcripts, input audio, generated tts audio, capability tests, and blind listening.

the tts listening page groups hidden models by sentence. ratings save in browser storage after each choice. `submit and reveal winner` requires every clip, then reveals the models and score table. `clear saved ratings` starts again.

## run from scratch

requirements: 64-bit windows, linux, or macos supported by the selected native runtimes; python 3.11 or 3.12; [uv](https://docs.astral.sh/uv/); internet access; enough disk for weights and isolated environments.

```bash
uv sync
uv run python scripts/download_data.py
uv run python scripts/download_models.py
uv run python scripts/run_all.py --fresh
```

`.venv` contains the controller and reporting tools. `.venvs/<runtime>` contains isolated inference stacks where model dependencies conflict. the download script creates both. compatible adapters share a runtime environment.

download commands are idempotent. valid files are checked and reused. missing files are downloaded. checksum changes stop the run. use `--force` to replace cached files.

```bash
uv run python scripts/download_data.py --dry-run
uv run python scripts/download_models.py --dry-run
uv run python scripts/download_models.py --model whisper_small_en --force
```

## benchmark commands

short check of every adapter:

```bash
uv run python scripts/run_all.py --smoke
```

resume a stopped full run without deleting successful jobs:

```bash
uv run python scripts/run_all.py --resume
```

show the job list:

```bash
uv run python scripts/run_all.py --dry-run
```

run one model:

```bash
uv run python scripts/run_all.py --kind asr --model moonshine_small --profile deployment --mode native
uv run python scripts/run_all.py --kind tts --model piper --profile deployment
```

rebuild tables, plots, and html from existing raw measurements:

```bash
uv run python scripts/run_all.py --plots-only
```

`--fresh` deletes old benchmark outputs before a full run. it does not delete data, models, environments, or caches. use `--resume` after interruption, not `--fresh`.

## method

each configuration runs in a fresh child process and its registered uv environment. load, warmup, and measured inference are separated. resource use is sampled during load and inference.

asr uses the same 16 khz mono files for every model: clean, fast, technical, speaker variation, deterministic noise, and long speech. native, silero vad, controlled-thread, and supported streaming paths are separate jobs.

tts uses the same six texts. each adapter uses its registered english voice and runtime defaults. supported voice, speed, cloning, pronunciation, normalization, streaming, noise, and expression controls run after generic measurement.

the deployment profile uses each runtime's normal thread choice. the controlled profile requests four threads where supported. exact scopes, repetitions, vad values, seed, memory limit, and ranking weights are in `config/benchmark.json`.

## outputs

- `outputs/master.json`: full result
- `outputs/processed/summary.csv`: model totals
- `outputs/processed/per_test.csv`: case-level rows
- `outputs/processed/rankings.csv`: four ranking profiles
- `outputs/processed/capabilities.csv`: feature results
- `outputs/processed/human_ratings.csv`: included blind ratings
- `outputs/processed/human_ratings_summary.csv`: model rating averages
- `outputs/audio/`: blind tts clips, feature-test clips, and copied asr inputs
- `outputs/plots/`: standalone charts
- `outputs/review/index.html`: dashboard

model weights, source datasets, environments, caches, and raw worker files are not committed. the published dashboard needs only committed files under `outputs/`.

## metrics

- wer and cer: transcription error rates; lower is better
- rtf: compute time divided by audio duration; below 1 is faster than real time
- first audio: delay before tts returns playable sound
- maximum rss: largest process memory measured in any case
- average rss: mean sampled process memory
- evaluator wer: tts intelligibility proxy, not naturalness

## registries

- `config/models.json`: model artifacts, runtimes, licenses, voices, and features
- `config/data.json`: asr sources and revisions
- `config/tts_texts.json`: generic tts cases
- `config/special_tests.json`: model-specific controls
- `config/benchmark.json`: profiles, scopes, vad, repetitions, and scores

## checks

```bash
uv run --frozen ruff check .
uv run --frozen ruff format --check .
uv run --frozen pytest
```

## limits

one whisper model supplies the tts intelligibility proxy. one listener supplied the included blind ratings. pseudo-streaming does not make an offline model native streaming. cpu scheduling, power policy, and temperature affect timing. licenses differ by runtime, weights, voice, and dataset; check `config/models.json` and the linked model cards before deployment.
