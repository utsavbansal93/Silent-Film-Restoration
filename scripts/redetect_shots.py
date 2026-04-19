"""Re-run ONLY shot detection for an existing run, using the current S01 config.
Reuses cached intertitle_frames + damage + motion from the existing probe_report.json,
so we avoid the 15-minute EAST rerun and the ~3-minute damage+flow recomputation.

Usage:
  python -m scripts.redetect_shots --run-dir runs/<name> [--config configs/modern_smooth.yaml]

Writes back to probe_report.json with an updated shot_boundaries list and
stores a shot_detection_config block for auditability.
"""
from __future__ import annotations

import json
from pathlib import Path

import click

from pipeline.common.config import load_config
from pipeline.common.logging import get_logger
from pipeline.common.paths import runs_dir
from pipeline.stages.s01_probe import detect_shots


@click.command()
@click.option("--run-dir", "run_dir", default=None,
              help="Run directory (default: newest run with s01 output).")
@click.option("--config", "cfg_path", default="configs/modern_smooth.yaml",
              help="Config for the new detection settings.")
def main(run_dir: str | None, cfg_path: str):
    cfg = load_config(cfg_path)

    if run_dir:
        run = Path(run_dir)
    else:
        candidates = sorted(runs_dir().iterdir(), reverse=True)
        run = next(
            (c for c in candidates if any(d.name.startswith("s01_") for d in c.iterdir())),
            None,
        )
        if run is None:
            raise RuntimeError("No run with s01 output found.")

    logger = get_logger("redetect_shots", run)

    s00 = next(d for d in run.iterdir() if d.name.startswith("s00_"))
    s01 = next(d for d in run.iterdir() if d.name.startswith("s01_"))
    frames_dir = s00 / "frames_raw"
    report_path = s01 / "probe_report.json"
    if not report_path.exists():
        raise RuntimeError(f"probe_report.json missing at {report_path}")

    report = json.loads(report_path.read_text())
    logger.info("Existing report: %d shots, %d intertitle frames",
                report.get("shot_count", 0), report.get("intertitle_count", 0))

    new_shots = detect_shots(frames_dir, cfg.s01_probe.shot_detection, logger)

    # Preserve everything else (damage, motion_magnitude, intertitle_frames, frame_count).
    report["shot_boundaries"] = new_shots
    report["shot_count"] = len(new_shots)
    report["shot_detection_config"] = cfg.s01_probe.shot_detection.model_dump(mode="json")

    report_path.write_text(json.dumps(report, indent=2))
    logger.info("Updated %s: %d shots (was %s).",
                report_path, len(new_shots), report.get("shot_count", "?"))


if __name__ == "__main__":
    main()
