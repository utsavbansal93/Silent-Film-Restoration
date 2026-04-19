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
    # Offset in seconds into the source file. ffmpeg -ss is applied at
    # extraction time so no re-encode is needed; source stays pristine.
    source_start_time_s: float = Field(default=0.0, ge=0.0)


class ShotDetectionCfg(BaseModel):
    # content      → ContentDetector only (fast, misses gradual cuts)
    # adaptive     → AdaptiveDetector only (good for gradual cuts)
    # content_plus_adaptive → union of both detectors (most coverage; default for silent film)
    method: Literal["content", "adaptive", "content_plus_adaptive"] = "content_plus_adaptive"
    threshold: float = 15.0                  # ContentDetector threshold; 27 is default, 15 catches subtler cuts
    adaptive_threshold: float = 3.0          # AdaptiveDetector threshold
    min_scene_len: int = 10                  # minimum frames between detected cuts (prevents duplicates)


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


class IntertitleCardCfg(BaseModel):
    id: str                                   # e.g. "C1"
    orig_start_frame: int = Field(ge=0)       # inclusive, 0-indexed into the S03 stream
    orig_end_frame: int = Field(ge=0)         # inclusive
    label: str = ""                           # freeform notes


class S04Cfg(BaseModel):
    cards: list[IntertitleCardCfg] = []       # empty = no-op passthrough (identity copy)


class S03Cfg(BaseModel):
    # hist_match = per-frame histogram matched against a rolling reference
    #   (median of nearby frames within the same shot). Primary — 91% flicker
    #   reduction on shot 21+22 sample, subsumes mean normalisation.
    # mean_norm = 1st-moment-only rescale. Fast, 82% on the same sample.
    # ffmpeg = ffmpeg -vf deflicker=size=5:mode=am. Baseline, 51%.
    # passthrough = copy frames unchanged.
    method: Literal["hist_match", "mean_norm", "ffmpeg", "passthrough"] = "hist_match"
    window: int = Field(default=25, ge=3, le=200)
    per_shot: bool = True
    skip_intertitles: bool = True


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
    s03_deflicker: S03Cfg = S03Cfg()
    s04_intertitle_extract: S04Cfg = S04Cfg()

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
            "s03": self.s03_deflicker,
            "s04": self.s04_intertitle_extract,
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
