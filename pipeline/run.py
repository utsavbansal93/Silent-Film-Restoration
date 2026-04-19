"""Top-level orchestrator. Runs configured stages in order against a single run dir."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import click

from pipeline.common.config import load_config
from pipeline.common.logging import get_logger
from pipeline.common.paths import new_run_dir

STAGE_MODULES = {
    "s00": "pipeline.stages.s00_ingest",
    "s01": "pipeline.stages.s01_probe",
    "s02": "pipeline.stages.s02_stabilise",
    "s03": "pipeline.stages.s03_deflicker",
    "s04": "pipeline.stages.s04_intertitle_extract",
    "s05": "pipeline.stages.s05_damage_classify",
}


@click.command()
@click.option("--config", "cfg_path", required=True)
def main(cfg_path: str) -> None:
    cfg = load_config(cfg_path)
    run_path = new_run_dir(cfg.profile_name)
    logger = get_logger("run", run_path)
    logger.info("Pipeline run starting; profile=%s; run_dir=%s", cfg.profile_name, run_path)

    for stage in cfg.run.stages_to_run:
        mod = STAGE_MODULES.get(stage)
        if not mod:
            logger.error("Unknown stage: %s — skipping", stage)
            continue
        logger.info("===== %s =====", stage)
        cmd = [sys.executable, "-m", mod, "--config", cfg_path, "--run-dir", str(run_path)]
        r = subprocess.run(cmd)
        if r.returncode != 0:
            logger.error("Stage %s failed (exit %d)", stage, r.returncode)
            sys.exit(r.returncode)
    logger.info("All stages complete. Run dir: %s", run_path)


if __name__ == "__main__":
    main()
