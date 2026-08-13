import importlib.metadata
import os
import platform
import random
import sys
import threading
import time
from contextlib import suppress
from dataclasses import dataclass, field

import numpy as np
import psutil


def environment():
    vm = psutil.virtual_memory()
    freq = psutil.cpu_freq()
    packages = {}
    for name in [
        "moonshine-voice",
        "pywhispercpp",
        "sherpa-onnx",
        "pocket-tts",
        "kokoro",
        "chatterbox-tts",
        "kittentts",
        "piper-tts",
        "onnxruntime",
        "torch",
    ]:
        with suppress(importlib.metadata.PackageNotFoundError):
            packages[name] = importlib.metadata.version(name)
    return {
        "platform": platform.platform(),
        "os": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version,
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "cpu_frequency_mhz": None if freq is None else freq.current,
        "ram_bytes": vm.total,
        "packages": packages,
        "thread_env": {
            key: os.environ.get(key)
            for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]
        },
    }


def set_threads(count):
    if count <= 0:
        return
    value = str(count)
    for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
        os.environ[key] = value
    try:
        import torch

        torch.set_num_threads(count)
        torch.set_num_interop_threads(1)
    except (ImportError, RuntimeError):
        pass


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def suppress_native_dialogs():
    if os.name != "nt":
        return
    import ctypes

    ctypes.windll.kernel32.SetErrorMode(0x0002 | 0x8000)


@dataclass
class ResourceSampler:
    interval: float = 0.02
    process: psutil.Process = field(default_factory=psutil.Process)
    samples: list = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None
    _cpu_start: float = 0.0

    def start(self):
        times = self.process.cpu_times()
        self._cpu_start = times.user + times.system
        self.process.cpu_percent(None)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            try:
                memory = self.process.memory_info().rss
                self.samples.append(
                    {
                        "t": time.perf_counter(),
                        "rss_bytes": memory,
                        "cpu_percent": self.process.cpu_percent(None),
                        "threads": self.process.num_threads(),
                    }
                )
            except psutil.Error:
                break
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=max(1.0, self.interval * 5))
        return self.summary()

    def summary(self):
        if not self.samples:
            return {}
        rss = [sample["rss_bytes"] for sample in self.samples]
        cpu = [sample["cpu_percent"] for sample in self.samples]
        threads = [sample["threads"] for sample in self.samples]
        times = self.process.cpu_times()
        return {
            "peak_rss_bytes": max(rss),
            "average_rss_bytes": sum(rss) / len(rss),
            "peak_cpu_percent": max(cpu),
            "average_cpu_percent": sum(cpu) / len(cpu),
            "peak_threads": max(threads),
            "cpu_time_s": times.user + times.system - self._cpu_start,
            "samples": len(self.samples),
        }


def directory_size(path):
    seen = set()
    total = 0
    for item in path.rglob("*"):
        if not item.is_file() or item.is_symlink():
            continue
        stat = item.stat()
        identity = (stat.st_dev, stat.st_ino)
        if identity not in seen:
            seen.add(identity)
            total += stat.st_size
    return total
