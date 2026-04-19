from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np


def preprocess(img_path: Path) -> np.ndarray:
    """Grayscale → denoise → deskew. Returns uint8 grayscale array for EasyOCR."""
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    denoised = cv2.fastNlMeansDenoising(img, h=10)
    angle = _estimate_skew(denoised)
    if abs(angle) > 0.5:
        return _rotate(denoised, -angle)
    return denoised


def _estimate_skew(gray: np.ndarray) -> float:
    """Return dominant text-line skew angle in degrees (positive = clockwise)."""
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, math.pi / 180, threshold=100)
    if lines is None:
        return 0.0
    angles: list[float] = []
    for line in lines[:20]:
        rho, theta = line[0]
        angle_deg = math.degrees(theta) - 90.0
        if abs(angle_deg) < 15:
            angles.append(angle_deg)
    return float(np.median(angles)) if angles else 0.0


def _rotate(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
