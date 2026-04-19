"""Standalone: compute pre/post shake metric on an existing run's
S00 raw + S02 stabilised frames, write shake_metric.json, run auto-QC.
Skips re-running S02 — use when only the metric is missing.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import click

from pipeline.common.logging import get_logger
from pipeline.common.paths import runs_dir
from pipeline.stages.s02_stabilise import frame_to_frame_translation_rms


@click.command()
@click.option("--run-dir", "run_dir", default=None)
def main(run_dir: str | None):
    if run_dir:
        run = Path(run_dir)
    else:
        candidates = sorted(runs_dir().iterdir(), reverse=True)
        run = next(
            (c for c in candidates if (c / "s02_stabilise" / "frames_stabilised").exists()),
            None,
        )
        if run is None:
            raise RuntimeError("No run with s02 stabilised output found.")

    logger = get_logger("shake_metric", run)
    s00 = next(d for d in run.iterdir() if d.name.startswith("s00_"))
    s02 = next(d for d in run.iterdir() if d.name.startswith("s02_"))

    raw_paths = sorted((s00 / "frames_raw").glob("*.png"))
    stab_paths = sorted((s02 / "frames_stabilised").glob("*.png"))
    logger.info("raw=%d stab=%d", len(raw_paths), len(stab_paths))
    if len(raw_paths) != len(stab_paths):
        logger.warning("frame count mismatch: raw=%d stab=%d", len(raw_paths), len(stab_paths))

    t0 = time.perf_counter()
    pre = frame_to_frame_translation_rms(raw_paths)
    t1 = time.perf_counter()
    logger.info("pre_shake_rms=%.4f  (%.1fs)", pre, t1 - t0)
    post = frame_to_frame_translation_rms(stab_paths)
    t2 = time.perf_counter()
    logger.info("post_shake_rms=%.4f (%.1fs)", post, t2 - t1)

    shake = {
        "method": "skimage_phase_corr",
        "pre_shake_rms": pre,
        "post_shake_rms": post,
        "reduction_pct": (1 - post / pre) * 100 if pre > 0 else 0.0,
    }
    out = s02 / "shake_metric.json"
    out.write_text(json.dumps(shake, indent=2))
    logger.info("wrote %s", out)
    print(f"\npre={pre:.4f}  post={post:.4f}  reduction={shake['reduction_pct']:.1f}%")


if __name__ == "__main__":
    main()
