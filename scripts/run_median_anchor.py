"""Run only the s02e_median second-pass variant on an existing run that
already has s02a_default output. Appends to s02_variants_comparison.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import click

from pipeline.common.logging import get_logger
from pipeline.common.paths import runs_dir
from scripts.compare_s02_variants import _run_median_anchor_second_pass


@click.command()
@click.option("--run-dir", "run_dir", default=None)
@click.option("--window", default=11, help="Temporal median window (symmetric; 11 = ±5).")
def main(run_dir: str | None, window: int):
    if run_dir:
        run = Path(run_dir)
    else:
        candidates = sorted(runs_dir().iterdir(), reverse=True)
        run = next((c for c in candidates if (c / "s02a_default").exists()), None)
        if run is None:
            raise RuntimeError("No run with s02a_default found.")
    logger = get_logger("s02e_runner", run)
    result = _run_median_anchor_second_pass(run, window, logger)

    # Append to comparison JSON if present.
    summary_path = run / "s02_variants_comparison.json"
    if summary_path.exists():
        existing = json.loads(summary_path.read_text())
        existing = [r for r in existing if r.get("variant") != "s02e_median"]
        existing.append(result)
        summary_path.write_text(json.dumps(existing, indent=2))

    print()
    print(f"{'variant':<20} {'pre':>7} {'post':>7} {'reduce':>8} {'wall_s':>8}")
    print(f"{result['variant']:<20} "
          f"{result['pre_shake_rms']:>7.3f} "
          f"{result['post_shake_rms']:>7.3f} "
          f"{result['reduction_pct']:>7.1f}% "
          f"{result['wall_time_s']:>7.1f}s")


if __name__ == "__main__":
    main()
