"""End-to-end smoke test on the 30-second fixture.

Skipped automatically if the fixture isn't present (run `just fetch-source
&& just regenerate-fixture` first).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from pipeline.common.paths import fixture_path, project_root

pytestmark = pytest.mark.integration


def _have_fixture() -> bool:
    return fixture_path().exists()


@pytest.mark.skipif(not _have_fixture(), reason="fixture missing — run `just regenerate-fixture`")
def test_s00_through_s02_on_fixture(tmp_path):
    cfg = project_root() / "configs" / "test_30sec.yaml"
    cmd = [sys.executable, "-m", "pipeline.run", "--config", str(cfg)]
    r = subprocess.run(cmd, cwd=project_root(), capture_output=True, text=True)
    assert r.returncode == 0, f"Pipeline failed:\n{r.stderr}"

    # Find the run dir we just created.
    runs = sorted((project_root() / "runs").glob("*_test_30sec"), reverse=True)
    assert runs, "No test_30sec run directory produced"
    run = runs[0]

    # Expect s00, s01, s02 subdirs with their outputs.
    s00 = next(run.glob("s00_*"))
    assert (s00 / "metadata.json").exists()
    assert (s00 / "bench.json").exists()
    assert list((s00 / "frames_raw").glob("*.png")), "S00 produced no frames"

    s01 = next(run.glob("s01_*"))
    assert (s01 / "probe_report.json").exists()
    assert (s01 / "bench.json").exists()

    s02 = next(run.glob("s02_*"))
    assert (s02 / "shake_metric.json").exists()
    assert (s02 / "bench.json").exists()
