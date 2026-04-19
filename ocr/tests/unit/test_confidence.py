import math

from ocr.common.confidence import aggregate, is_devanagari, split_by_script


def test_is_devanagari_true():
    assert is_devanagari("नमस्ते")
    assert is_devanagari("hello नमस्ते")


def test_is_devanagari_false():
    assert not is_devanagari("hello world")
    assert not is_devanagari("")
    assert not is_devanagari("Lankadahan.")


def test_split_by_script_separates():
    results = [([], "hello", 0.9), ([], "नमस्ते", 0.7), ([], "world", 0.8)]
    latin, deva = split_by_script(results)
    assert len(latin) == 2
    assert len(deva) == 1
    assert deva[0][0] == "नमस्ते"


def test_split_skips_empty():
    results = [([], "", 0.9), ([], "  ", 0.5)]
    latin, deva = split_by_script(results)
    assert latin == []
    assert deva == []


def test_aggregate_empty():
    r = aggregate([])
    assert r["text"] == ""
    assert r["confidence"] == 0.0


def test_aggregate_single():
    r = aggregate([("hello", 0.9)])
    assert r["text"] == "hello"
    assert abs(r["confidence"] - 0.9) < 1e-6


def test_aggregate_mean_confidence():
    r = aggregate([("a", 0.8), ("b", 0.6)])
    assert abs(r["confidence"] - 0.7) < 1e-4


def test_aggregate_no_nan_inf():
    r = aggregate([("x", 1.0), ("y", 0.0)])
    assert not math.isnan(r["confidence"])
    assert not math.isinf(r["confidence"])


def test_confidence_in_unit_interval():
    r = aggregate([("a", 0.0), ("b", 1.0)])
    assert 0.0 <= r["confidence"] <= 1.0


def test_aggregate_joins_text():
    r = aggregate([("Bravo", 0.9), ("!", 0.85)])
    assert "Bravo" in r["text"]
    assert "!" in r["text"]
