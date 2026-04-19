"""YAML config loader + validator (pydantic). Fails loudly on missing/bad fields."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

from pipeline.common.hashing import sha256_json


class SourceCfg(BaseModel):
    path: str
    sha256: str | None = None


class RunCfg(BaseModel):
    output_root: str = "runs/"
    stages_to_run: list[str] = Field(default_factory=lambda: ["s00", "s01", "s02"])


class S00Cfg(BaseModel):
    output_format: Literal["png", "tiff"] = "png"
    preserve_audio: bool = False
    parallel_workers: int = Field(default=1, ge=1, le=16)


class ShotDetectionCfg(BaseModel):
    method: Literal["pyscenedetect"] = "pyscenedetect"
    threshold: float = 27.0


class DamageHeuristicsCfg(BaseModel):
    laplacian_blur_threshold: float = 100.0
    dark_frame_threshold: float = 0.02


class IntertitleDetectionCfg(BaseModel):
    enabled: bool = True
    method: Literal["east"] = "east"
    model_path: str = "models/frozen_east_text_detection.pb"
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    # Fraction of frame area detected as text to flag as intertitle.
    area_threshold: float = Field(default=0.08, ge=0.0, le=1.0)
    # Downscale frames for speed; 320/640 are EAST-native resolutions.
    resize_width: int = 640
    resize_height: int = 384


class S01Cfg(BaseModel):
    shot_detection: ShotDetectionCfg = ShotDetectionCfg()
    damage_heuristics: DamageHeuristicsCfg = DamageHeuristicsCfg()
    intertitle_detection: IntertitleDetectionCfg = IntertitleDetectionCfg()


class S02Cfg(BaseModel):
    # skimage_phase_corr: FFT-based sub-pixel frame-to-frame alignment.
    #   Designed for film weave (hand-crank gate jitter). Robust to dirt/grain.
    # opencv_features: feature-tracking (broken on tripod-stable film-weave sources).
    # passthrough: copy frames unchanged; downstream stages proceed as if S02 is a no-op.
    method: Literal["skimage_phase_corr", "opencv_features", "passthrough"] = "skimage_phase_corr"
    # Wide smoothing window for weave removal: weave is high-freq jitter; genuine
    # pans/motion are multi-second low-freq. Wide window separates them well.
    smoothing: int = Field(default=25, ge=3, le=200)
    # Sub-pixel upsampling for phase_cross_correlation. 10 → 0.1 px precision.
    upsample_factor: int = Field(default=10, ge=1, le=100)
    per_shot: bool = True
    crop: Literal["keep", "zoom"] = "keep"
    skip_intertitles: bool = True


class PipelineConfig(BaseModel):
    profile_name: str
    source: SourceCfg
    run: RunCfg = RunCfg()
    s00_ingest: S00Cfg = S00Cfg()
    s01_probe: S01Cfg = S01Cfg()
    s02_stabilise: S02Cfg = S02Cfg()

    @field_validator("profile_name")
    @classmethod
    def _profile_name_shape(cls, v: str) -> str:
        if not v or not all(c.isalnum() or c in "_-" for c in v):
            raise ValueError("profile_name must be non-empty alphanumeric + _/-")
        return v

    def section_hash(self, stage_id: str) -> str:
        mapping = {
            "s00": self.s00_ingest,
            "s01": self.s01_probe,
            "s02": self.s02_stabilise,
        }
        if stage_id not in mapping:
            raise KeyError(f"Unknown stage id: {stage_id}")
        return sha256_json(mapping[stage_id].model_dump(mode="json"))


def load_config(path: str | Path) -> PipelineConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open() as f:
        data = yaml.safe_load(f)
    return PipelineConfig.model_validate(data)
