from __future__ import annotations

import math

# Unicode Devanagari block: U+0900–U+097F
_DEVA_LO = 0x0900
_DEVA_HI = 0x097F


def is_devanagari(text: str) -> bool:
    return any(_DEVA_LO <= ord(c) <= _DEVA_HI for c in text)


def split_by_script(
    results: list[tuple[list, str, float]],
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """Split EasyOCR results into (latin_items, devanagari_items) as (text, conf) pairs."""
    latin: list[tuple[str, float]] = []
    deva: list[tuple[str, float]] = []
    for _bbox, text, conf in results:
        text = text.strip()
        if not text:
            continue
        if is_devanagari(text):
            deva.append((text, float(conf)))
        else:
            latin.append((text, float(conf)))
    return latin, deva


def aggregate(items: list[tuple[str, float]]) -> dict:
    """Combine (text, confidence) pairs into a single {text, confidence} dict."""
    if not items:
        return {"text": "", "confidence": 0.0}
    text = " ".join(t for t, _ in items)
    confs = [c for _, c in items]
    mean_conf = sum(confs) / len(confs)
    if math.isnan(mean_conf) or math.isinf(mean_conf):
        mean_conf = 0.0
    return {"text": text, "confidence": round(mean_conf, 4)}
