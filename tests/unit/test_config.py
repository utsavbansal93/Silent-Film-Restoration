import pytest

from pipeline.common.config import PipelineConfig, load_config


def test_load_modern_smooth():
    cfg = load_config("configs/modern_smooth.yaml")
    assert cfg.profile_name == "modern_smooth"
    assert cfg.s01_probe.intertitle_detection.enabled
    assert cfg.s02_stabilise.skip_intertitles


def test_load_test_30sec():
    cfg = load_config("configs/test_30sec.yaml")
    assert cfg.profile_name == "test_30sec"


def test_section_hash_is_deterministic():
    cfg = load_config("configs/modern_smooth.yaml")
    a = cfg.section_hash("s02")
    b = cfg.section_hash("s02")
    assert a == b and len(a) == 64


def test_section_hash_differs_across_stages():
    cfg = load_config("configs/modern_smooth.yaml")
    assert cfg.section_hash("s00") != cfg.section_hash("s02")


def test_bad_profile_name_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("profile_name: 'bad name with space'\nsource:\n  path: x\n")
    with pytest.raises(Exception):
        load_config(p)


def test_unknown_stage_id_raises():
    cfg = load_config("configs/modern_smooth.yaml")
    with pytest.raises(KeyError):
        cfg.section_hash("s99")
