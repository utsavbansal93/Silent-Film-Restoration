"""Per-stage bench.json writer (brief §5a)."""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

import psutil

from pipeline.common.device import device_label


@dataclass
class Bench:
    stage: str
    device: str = field(default_factory=device_label)
    frames_processed: int = 0
    wall_time_seconds: float = 0.0
    frames_per_second: float = 0.0
    peak_memory_mb: float = 0.0
    batch_size: int | None = None
    model_version: str | None = None
    config_hash: str = ""
    input_hash: str = ""
    notes: str = ""


@contextmanager
def bench_run(stage_id: str, config_hash: str, out_dir: Path):
    """Usage:
        with bench_run("s00_ingest", cfg.section_hash("s00"), out_dir) as b:
            ... process frames, set b.frames_processed, b.input_hash ...
    """
    b = Bench(stage=stage_id, config_hash=config_hash)
    proc = psutil.Process(os.getpid())
    start = time.perf_counter()
    peak = proc.memory_info().rss

    def sample_peak():
        nonlocal peak
        rss = proc.memory_info().rss
        if rss > peak:
            peak = rss

    b._sample = sample_peak  # type: ignore[attr-defined]
    try:
        yield b
    finally:
        b.wall_time_seconds = time.perf_counter() - start
        if b.wall_time_seconds > 0 and b.frames_processed > 0:
            b.frames_per_second = b.frames_processed / b.wall_time_seconds
        sample_peak()
        b.peak_memory_mb = peak / (1024 * 1024)
        out_path = out_dir / "bench.json"
        payload = {k: v for k, v in asdict(b).items() if not k.startswith("_")}
        out_path.write_text(json.dumps(payload, indent=2))
