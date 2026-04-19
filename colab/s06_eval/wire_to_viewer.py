"""Place VapourSynth eval outputs into a viewer-discoverable stage dir.

Target layout:
  runs/canonical-base/s06_vapoursynth_eval/frames_cleaned/
    00000001.png  -> hardlink to s05 frames_classified/00000001.png (outside-shot)
    ...
    00000125.png  -> hardlink to vapoursynth/shot_00/out_00000125.png  (inside shot 0)
    ...

Every frame position exists so the viewer can scrub the full 6,890-frame stream
and A/B against S05. Only frames inside {shot 0, shot 19, shot 21} are actually
VapourSynth-processed; the rest fall through to the S05 output unchanged.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
S05_DIR = ROOT / "runs" / "canonical-base" / "s05_damage_classify" / "frames_classified"
VS_OUT  = ROOT / "runs" / "s06-eval" / "output" / "vapoursynth"
OUT_DIR = ROOT / "runs" / "canonical-base" / "s06_vapoursynth_eval" / "frames_cleaned"

# (shot_id, start, end inclusive) — must match run_vapoursynth.py's SHOTS.
SHOTS = [(0, 1, 125), (19, 1450, 1538), (21, 1561, 1768)]


def main() -> None:
    # Build the override map: frame-name -> source path.
    overrides: dict[str, Path] = {}
    for sid, start, end in SHOTS:
        shot_dir = VS_OUT / f"shot_{sid:02d}"
        for n in range(start, end + 1):
            src = shot_dir / f"out_{n:08d}.png"
            if src.exists():
                overrides[f"{n:08d}.png"] = src
            else:
                print(f"  MISSING: {src}")

    print(f"override count: {len(overrides)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Clear any stale links.
    for p in OUT_DIR.iterdir():
        p.unlink()

    s05_frames = sorted(S05_DIR.glob("*.png"))
    created = passthrough = 0
    for f in s05_frames:
        dst = OUT_DIR / f.name
        src = overrides.get(f.name, f)
        os.link(src, dst)
        if f.name in overrides:
            created += 1
        else:
            passthrough += 1

    print(f"VS-cleaned links:  {created}")
    print(f"S05 passthrough:   {passthrough}")
    print(f"Total in {OUT_DIR.relative_to(ROOT)}: {created + passthrough}")


if __name__ == "__main__":
    main()
