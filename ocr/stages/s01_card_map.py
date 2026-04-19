from __future__ import annotations

import json
from pathlib import Path


def build_card_map(s04_dir: Path) -> list[dict]:
    """Read S04 intertitles/ structure. Returns list of card dicts with resolved paths."""
    intertitles_dir = s04_dir / "intertitles"
    if not intertitles_dir.exists():
        raise FileNotFoundError(f"S04 intertitles dir not found: {intertitles_dir}")

    cards: list[dict] = []
    for card_dir in sorted(intertitles_dir.iterdir()):
        if not card_dir.is_dir():
            continue
        meta_path = card_dir / "meta.json"
        rep_path = card_dir / "representative.png"
        if not meta_path.exists() or not rep_path.exists():
            continue

        meta = json.loads(meta_path.read_text())
        raw_id = meta["id"].lower()          # "C1" → "c1"
        card_num = int(raw_id.lstrip("c"))   # "c1" → 1
        card_id = f"card_{card_num:03d}"

        frames_dir = card_dir / "frames"
        all_frames = sorted(frames_dir.glob("*.png")) if frames_dir.exists() else []

        cards.append({
            "card_id": card_id,
            "s04_id": meta["id"],
            "label": meta.get("label", ""),
            "representative": rep_path,
            "frames_dir": frames_dir,
            "all_frames": all_frames,
            "frame_count": meta.get("frame_count", len(all_frames)),
            "duration_s": meta.get("duration_s", 0.0),
        })

    return cards
