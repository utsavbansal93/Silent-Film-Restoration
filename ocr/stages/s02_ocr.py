from __future__ import annotations

import logging
from pathlib import Path

import easyocr

from ocr.common.confidence import aggregate, split_by_script
from ocr.common.image_prep import preprocess

logger = logging.getLogger(__name__)

_EN_CONF_FLOOR = 0.85
_DEVA_CONF_FLOOR = 0.55
_RETRY_THRESHOLD = 0.5  # if both fields below this, try one fallback frame


def make_reader() -> easyocr.Reader:
    """Initialise EasyOCR with Hindi + English models (downloads ~500 MB on first run)."""
    logger.info("Initialising EasyOCR reader ['hi', 'en'] — first run downloads models")
    return easyocr.Reader(["hi", "en"], gpu=False, verbose=False)


def ocr_card(card: dict, reader: easyocr.Reader) -> dict:
    """OCR one card. Uses representative.png; single fallback frame if both confidences are low."""
    rep: Path = card["representative"]
    english, hindi, source_frames = _ocr_image(rep, reader)

    # Single fallback: only if both fields look empty/noise
    if (
        english["confidence"] < _RETRY_THRESHOLD
        and hindi["confidence"] < _RETRY_THRESHOLD
        and card["all_frames"]
    ):
        frames = card["all_frames"]
        mid = frames[len(frames) // 2]
        if mid.resolve() != rep.resolve():
            logger.info("  %s: low confidence — retrying with mid-card frame %s", card["card_id"], mid.name)
            en2, hi2, _ = _ocr_image(mid, reader)
            if en2["confidence"] + hi2["confidence"] > english["confidence"] + hindi["confidence"]:
                english, hindi = en2, hi2
                source_frames = [mid.name]
                logger.info("  %s: fallback improved confidence", card["card_id"])

    notes: list[str] = []
    if not english["text"] and not hindi["text"]:
        notes.append("no text detected")
    else:
        if english["confidence"] < _EN_CONF_FLOOR and english["text"]:
            notes.append(f"low English confidence ({english['confidence']:.2f})")
        if hindi["confidence"] < _DEVA_CONF_FLOOR and hindi["text"]:
            notes.append(f"low Devanagari confidence ({hindi['confidence']:.2f})")

    return {
        "card_id": card["card_id"],
        "source_frames": source_frames,
        "marathi": {"text": "", "confidence": 0.0},
        "hindi": hindi,
        "english": english,
        "notes": "; ".join(notes),
    }


def _ocr_image(img_path: Path, reader: easyocr.Reader) -> tuple[dict, dict, list[str]]:
    img = preprocess(img_path)
    raw = reader.readtext(img, detail=1)
    latin, deva = split_by_script(raw)
    return aggregate(latin), aggregate(deva), [img_path.name]
