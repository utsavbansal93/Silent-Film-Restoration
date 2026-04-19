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


def stage_input_frames(run_dir: Path, stage_id: str) -> Path:
    """Return the frames directory a given stage should read from.

    Encodes the pipeline's DAG contract in one place so future stages don't
    have to hard-code their predecessor's path:

        S02 (stabilise)            ← S00 frames_raw/
        S03 (deflicker)            ← S02 frames_stabilised/
        S04 (intertitle_extract)   ← S03 frames_deflickered/
        S05+ (future stages)       ← S04 frames_movie/  (falls back to S03 if S04 never ran)
    """
    def _find(prefix: str) -> Path | None:
        for d in run_dir.iterdir():
            if d.is_dir() and d.name.startswith(prefix):
                return d
        return None

    if stage_id == "s02":
        s00 = _find("s00_")
        if s00 is None:
            raise FileNotFoundError(f"No s00_* under {run_dir}")
        return s00 / "frames_raw"
    if stage_id == "s03":
        s02 = _find("s02_")
        if s02 is None:
            raise FileNotFoundError(f"No s02_* under {run_dir}")
        return s02 / "frames_stabilised"
    if stage_id == "s04":
        s03 = _find("s03_")
        if s03 is None:
            raise FileNotFoundError(f"No s03_* under {run_dir}")
        return s03 / "frames_deflickered"
    # S05 and beyond: prefer S04's movie-only stream; fall back to S03 if S04 absent.
    s04 = _find("s04_")
    if s04 is not None:
        return s04 / "frames_movie"
    s03 = _find("s03_")
    if s03 is None:
        raise FileNotFoundError(f"No s03_* or s04_* under {run_dir}")
    return s03 / "frames_deflickered"
