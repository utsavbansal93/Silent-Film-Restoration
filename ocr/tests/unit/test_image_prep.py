import numpy as np
import pytest

from ocr.common.image_prep import _estimate_skew, _rotate


def test_rotate_preserves_shape():
    img = np.zeros((100, 200), dtype=np.uint8)
    assert _rotate(img, 5.0).shape == img.shape


def test_rotate_zero_is_noop():
    img = (np.random.default_rng(0).integers(0, 255, (80, 120)) * 1).astype(np.uint8)
    result = _rotate(img, 0.0)
    assert result.shape == img.shape


def test_estimate_skew_blank_image_returns_zero():
    img = np.zeros((100, 200), dtype=np.uint8)
    assert _estimate_skew(img) == 0.0


def test_estimate_skew_returns_float():
    img = np.zeros((100, 200), dtype=np.uint8)
    angle = _estimate_skew(img)
    assert isinstance(angle, float)
    assert abs(angle) <= 90.0
