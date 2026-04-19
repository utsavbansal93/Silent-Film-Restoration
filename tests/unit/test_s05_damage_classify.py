"""Unit tests for S05 damage_classify pure functions.

All tests use explicit S05Cfg values rather than defaults so threshold changes
don't silently break the test suite — each test documents the intent through
its threshold choices, not through reliance on default behaviour.
"""
import numpy as np
import pytest

from pipeline.common.config import S05Cfg, load_config
from pipeline.stages.s05_damage_classify import (
    FrameFeatures,
    ResolvedThresholds,
    classify_frame,
    compute_calibration,
    compute_rolling_brightness,
    resolve_thresholds,
)

# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

def test_s05_config_defaults():
    """S05Cfg with no YAML block → pydantic defaults applied correctly."""
    cfg = S05Cfg()
    assert cfg.method == "heuristic"
    assert cfg.workers == 0
    assert cfg.cat_c_laplacian_pct == 1.0
    assert cfg.cat_b_laplacian_pct == 5.0
    assert cfg.cat_c_laplacian_abs is None
    assert cfg.cat_b_laplacian_abs is None
    assert cfg.cat_c_mean_low == 5.0
    assert cfg.cat_c_mean_high == 240.0
    assert cfg.cat_c_extreme_ratio == 0.90
    assert cfg.cat_b_extreme_ratio == 0.30
    assert cfg.temporal_brightness_delta == 30.0


def test_s05_config_loads_from_yaml():
    """modern_smooth.yaml s05 block parses and values match what's declared."""
    cfg = load_config("configs/modern_smooth.yaml")
    s05 = cfg.s05_damage_classify
    assert s05.method == "heuristic"
    assert s05.cat_c_laplacian_pct == 1.0
    assert s05.cat_b_laplacian_pct == 5.0
    assert s05.cat_c_mean_high == 240.0
    assert s05.temporal_brightness_delta == 30.0


def test_s05_section_hash_deterministic():
    """section_hash('s05') is stable across calls and is 64 hex chars."""
    cfg = load_config("configs/modern_smooth.yaml")
    h1 = cfg.section_hash("s05")
    h2 = cfg.section_hash("s05")
    assert h1 == h2
    assert len(h1) == 64
    assert all(c in "0123456789abcdef" for c in h1)


# ---------------------------------------------------------------------------
# resolve_thresholds tests
# ---------------------------------------------------------------------------

def _feats(lap_vals: list[float], bright: float = 100.0) -> list[FrameFeatures]:
    return [
        FrameFeatures(laplacian_var=v, mean_brightness=bright,
                      dark_pixel_ratio=0.01, bright_pixel_ratio=0.01)
        for v in lap_vals
    ]


def test_resolve_thresholds_percentile_ordering():
    """cat_b_laplacian is always >= cat_c_laplacian after resolution."""
    lap_vals = list(range(1, 101))  # 100 values: 1..100
    feats = _feats(lap_vals)
    cfg = S05Cfg(cat_c_laplacian_pct=5.0, cat_b_laplacian_pct=20.0)
    t = resolve_thresholds(feats, cfg)
    assert t.cat_b_laplacian >= t.cat_c_laplacian


def test_resolve_thresholds_absolute_override():
    """Absolute override bypasses percentile computation."""
    lap_vals = list(range(1, 101))
    feats = _feats(lap_vals)
    cfg = S05Cfg(cat_c_laplacian_abs=10.0, cat_b_laplacian_abs=50.0)
    t = resolve_thresholds(feats, cfg)
    assert t.cat_c_laplacian == 10.0
    assert t.cat_b_laplacian == 50.0


# ---------------------------------------------------------------------------
# classify_frame tests (explicit thresholds, no defaults)
# ---------------------------------------------------------------------------

def _thresholds(
    cat_c_lap: float = 100.0,
    cat_b_lap: float = 300.0,
    cat_c_mean_low: float = 5.0,
    cat_c_mean_high: float = 240.0,
    cat_c_extreme: float = 0.90,
    cat_b_extreme: float = 0.30,
) -> ResolvedThresholds:
    return ResolvedThresholds(
        cat_c_laplacian=cat_c_lap,
        cat_b_laplacian=cat_b_lap,
        cat_c_mean_low=cat_c_mean_low,
        cat_c_mean_high=cat_c_mean_high,
        cat_c_extreme_ratio=cat_c_extreme,
        cat_b_extreme_ratio=cat_b_extreme,
    )


def _cfg(delta: float = 30.0) -> S05Cfg:
    return S05Cfg(temporal_brightness_delta=delta)


def test_classify_cat_a_sharp_normally_lit():
    """Sharp, well-lit frame → cat_a, reason None."""
    feats = FrameFeatures(
        laplacian_var=500.0, mean_brightness=110.0,
        dark_pixel_ratio=0.02, bright_pixel_ratio=0.01,
    )
    cat, reason = classify_frame(feats, _thresholds(), rolling_brightness_mean=112.0, cfg=_cfg())
    assert cat == "cat_a"
    assert reason is None


def test_classify_cat_c_near_black():
    """Mean brightness well below cat_c_mean_low → cat_c."""
    feats = FrameFeatures(
        laplacian_var=500.0, mean_brightness=2.0,
        dark_pixel_ratio=0.95, bright_pixel_ratio=0.0,
    )
    cat, reason = classify_frame(feats, _thresholds(cat_c_mean_low=5.0), None, _cfg())
    assert cat == "cat_c"
    assert reason is not None
    assert "mean_brightness" in reason


def test_classify_cat_c_low_laplacian():
    """Laplacian variance below cat_c threshold → cat_c."""
    feats = FrameFeatures(
        laplacian_var=50.0, mean_brightness=120.0,
        dark_pixel_ratio=0.01, bright_pixel_ratio=0.01,
    )
    cat, reason = classify_frame(feats, _thresholds(cat_c_lap=100.0), None, _cfg())
    assert cat == "cat_c"
    assert "laplacian_var" in reason
    assert "cat_c threshold" in reason


def test_classify_cat_b_moderate_laplacian():
    """Laplacian between cat_c and cat_b thresholds → cat_b."""
    feats = FrameFeatures(
        laplacian_var=200.0, mean_brightness=110.0,
        dark_pixel_ratio=0.02, bright_pixel_ratio=0.01,
    )
    # cat_c_lap=100, cat_b_lap=300 → 200 is in the cat_b range
    cat, reason = classify_frame(feats, _thresholds(cat_c_lap=100.0, cat_b_lap=300.0), None, _cfg())
    assert cat == "cat_b"
    assert "cat_b threshold" in reason


def test_classify_cat_b_temporal_upgrade():
    """Frame above all static thresholds but anomalous brightness → upgraded to cat_b."""
    feats = FrameFeatures(
        laplacian_var=500.0, mean_brightness=170.0,  # high but not extreme
        dark_pixel_ratio=0.01, bright_pixel_ratio=0.01,
    )
    # rolling mean = 120 → delta = 50 > 30 → upgrade
    cat, reason = classify_frame(
        feats, _thresholds(), rolling_brightness_mean=120.0, cfg=_cfg(delta=30.0)
    )
    assert cat == "cat_b"
    assert "brightness_delta" in reason


def test_classify_temporal_upgrade_disabled():
    """temporal_brightness_delta=0.0 disables temporal signal → cat_a even with large delta."""
    feats = FrameFeatures(
        laplacian_var=500.0, mean_brightness=200.0,
        dark_pixel_ratio=0.01, bright_pixel_ratio=0.01,
    )
    cat, reason = classify_frame(
        feats, _thresholds(), rolling_brightness_mean=100.0, cfg=_cfg(delta=0.0)
    )
    assert cat == "cat_a"
    assert reason is None


# ---------------------------------------------------------------------------
# Rolling brightness test
# ---------------------------------------------------------------------------

def test_rolling_brightness_resets_at_shot_boundaries():
    """Rolling mean is computed independently per shot — no leakage across cuts."""
    # 6 frames: shot A = [0..2], shot B = [3..5]
    names = ["f0", "f1", "f2", "f3", "f4", "f5"]
    feats_map = {
        "f0": FrameFeatures(laplacian_var=100.0, mean_brightness=10.0, dark_pixel_ratio=0.0, bright_pixel_ratio=0.0),
        "f1": FrameFeatures(laplacian_var=100.0, mean_brightness=12.0, dark_pixel_ratio=0.0, bright_pixel_ratio=0.0),
        "f2": FrameFeatures(laplacian_var=100.0, mean_brightness=11.0, dark_pixel_ratio=0.0, bright_pixel_ratio=0.0),
        "f3": FrameFeatures(laplacian_var=100.0, mean_brightness=200.0, dark_pixel_ratio=0.0, bright_pixel_ratio=0.0),
        "f4": FrameFeatures(laplacian_var=100.0, mean_brightness=205.0, dark_pixel_ratio=0.0, bright_pixel_ratio=0.0),
        "f5": FrameFeatures(laplacian_var=100.0, mean_brightness=202.0, dark_pixel_ratio=0.0, bright_pixel_ratio=0.0),
    }
    result = compute_rolling_brightness(names, feats_map, shot_boundaries=[3], window=5)
    # Shot A rolling means should be close to 10-12
    assert result["f0"] < 50.0
    assert result["f2"] < 50.0
    # Shot B rolling means should be close to 200-205
    assert result["f3"] > 150.0
    assert result["f5"] > 150.0


# ---------------------------------------------------------------------------
# Calibration structure test
# ---------------------------------------------------------------------------

def test_compute_calibration_structure():
    """compute_calibration returns expected keys and resolved_thresholds block."""
    feats = _feats(list(range(10, 110)))  # 100 values 10..109
    cfg = S05Cfg(cat_c_laplacian_pct=1.0, cat_b_laplacian_pct=5.0)
    thresholds = resolve_thresholds(feats, cfg)
    calib = compute_calibration(feats, thresholds)

    for feature in ("laplacian_var", "mean_brightness", "dark_pixel_ratio", "bright_pixel_ratio"):
        assert feature in calib
        for stat in ("min", "p5", "p25", "median", "p75", "p95", "max"):
            assert stat in calib[feature], f"Missing {stat} in {feature}"

    rt = calib["resolved_thresholds"]
    assert "cat_c_laplacian" in rt
    assert "cat_b_laplacian" in rt
    assert rt["cat_b_laplacian"] >= rt["cat_c_laplacian"]
