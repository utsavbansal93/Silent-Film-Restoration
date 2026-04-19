"""Path helpers. Environment-aware roots and timestamped run directories."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

ENV_VAR = "SILENT_FILM_ENV"


def env() -> str:
    return os.environ.get(ENV_VAR, "local").lower()


def project_root() -> Path:
    """Root dir for *this* project (Lanka Dahan). Same tree on every env; root differs."""
    e = env()
    if e == "colab":
        return Path("/content/drive/MyDrive/silent-film-restoration/lanka-dahan")
    if e == "kaggle":
        return Path("/kaggle/working/silent-film-restoration/lanka-dahan")
    return Path(__file__).resolve().parents[2]


def runs_dir() -> Path:
    d = project_root() / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def new_run_dir(profile_name: str) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    d = runs_dir() / f"{ts}_{profile_name}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def stage_dir(run_dir: Path, stage_id: str, name: str) -> Path:
    d = run_dir / f"{stage_id}_{name}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def source_path() -> Path:
    return project_root() / "source" / "lanka_dahan_1917.webm"


def fixture_path() -> Path:
    return project_root() / "tests" / "fixtures" / "test_clip_30sec.webm"


def models_dir() -> Path:
    d = project_root() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d
