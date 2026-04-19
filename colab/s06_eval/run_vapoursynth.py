"""Shot-level VapourSynth eval for S06 dirt_remove.

Pipeline per shot:
  imwri.Read (PNG sequence)
    -> RGB24 -> YUV (DeScratch operates on luma)
    -> rgvs.Clense (temporal median across 3 frames — removes moving dust)
    -> descratch.DeScratch (vertical scratch detect+remove)
    -> rgvs.RemoveGrain mode 17 (mild spatial denoise)
    -> back to RGB24
    -> imwri.Write (PNG sequence)

Produces side-by-side input/output mp4s per shot for viewer-playable A/B.

Requires native arm64: brew vapoursynth + vapoursynth-imwri + rdvs + rgvs + mvtools
+ source-built descratch. See JOURNAL D20 for install notes.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FRAMES_DIR = ROOT / "runs" / "canonical-base" / "s05_damage_classify" / "frames_classified"
OUT_ROOT = ROOT / "runs" / "s06-eval" / "output" / "vapoursynth"

# Three shots with distinct damage profiles. Boundaries from
# runs/canonical-base/s04_intertitle_extract/shot_boundaries_adjusted.json.
SHOTS = [
    {"id": 0,  "start": 1,    "end": 125,  "label": "opening (clean control)"},
    {"id": 19, "start": 1450, "end": 1538, "label": "vertical scratch (top-right to bottom-right)"},
    {"id": 21, "start": 1561, "end": 1768, "label": "longer shot, dust/dirt candidates"},
]

VSPIPE = "/opt/homebrew/bin/vspipe"

VS_SCRIPT = """
import os, vapoursynth as vs
core = vs.core

frames_dir = os.environ['VS_FRAMES_DIR']
pattern    = os.environ['VS_PATTERN']  # e.g. %08d.png
first_num  = int(os.environ['VS_FIRST'])
last_num   = int(os.environ['VS_LAST'])
out_dir    = os.environ['VS_OUT_DIR']

src = core.imwri.Read(
    [f"{frames_dir}/{pattern % n}" for n in range(first_num, last_num + 1)],
)

# YUV for DeScratch (operates on luma plane).
yuv = core.resize.Bicubic(src, format=vs.YUV444P8, matrix_s="709")

# 3-frame temporal median on luma+chroma via Clense (handles dust/specks).
clensed = core.rgvs.Clense(yuv)

# Vertical scratch detection + removal (Fizick / Mellbin).
# mindif=5: pixel-luma threshold for scratch-to-neighbour contrast.
# maxwidth=4: px width up to which a vertical run is considered a scratch.
# minlen=50: minimum vertical run length (in px) to flag as scratch.
# modey=1: process luma plane.
descratched = core.descratch.DeScratch(clensed, mindif=5, maxwidth=5, minlen=50, modey=1)

# Mild spatial denoise (mode 17 = "line-sensitive", preserves edges).
clean = core.rgvs.RemoveGrain(descratched, mode=17)

# Back to RGB24 for PNG output.
rgb = core.resize.Bicubic(clean, format=vs.RGB24, matrix_in_s="709")

core.imwri.Write(rgb, imgformat="PNG", filename=f"{out_dir}/out_%08d.png", firstnum=first_num).set_output()
"""


def process_shot(shot: dict) -> dict:
    sid, start, end = shot["id"], shot["start"], shot["end"]
    shot_out = OUT_ROOT / f"shot_{sid:02d}"
    shot_out.mkdir(parents=True, exist_ok=True)
    # Clear any stale outputs.
    for p in shot_out.glob("out_*.png"):
        p.unlink()

    script_path = shot_out / "_vs_script.py"
    script_path.write_text(VS_SCRIPT)

    env = {
        "VS_FRAMES_DIR": str(FRAMES_DIR),
        "VS_PATTERN": "%08d.png",
        "VS_FIRST": str(start),
        "VS_LAST": str(end),
        "VS_OUT_DIR": str(shot_out),
        "PATH": "/opt/homebrew/bin:/usr/bin:/bin",
    }
    t0 = time.time()
    proc = subprocess.run(
        [VSPIPE, "--progress", str(script_path), "."],   # "." = discard output; imwri.Write does the real write
        capture_output=True, text=True, env=env,
    )
    wall = time.time() - t0
    n_out = len(list(shot_out.glob("out_*.png")))
    info = {
        "shot_id": sid, "start": start, "end": end, "n_in": end - start + 1,
        "n_out": n_out, "wall_s": round(wall, 2),
        "fps": round(n_out / max(wall, 1e-6), 2),
        "returncode": proc.returncode,
        "stderr_tail": proc.stderr[-500:] if proc.stderr else "",
    }
    if proc.returncode != 0 or n_out == 0:
        print(f"\n[FAIL shot {sid}] stderr:\n{proc.stderr[-2000:]}")
    else:
        print(f"shot {sid:2d}: {n_out}/{end-start+1} frames cleaned in {wall:.1f}s ({n_out/wall:.1f} fps)")
    return info


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    for shot in SHOTS:
        print(f"--- shot {shot['id']:2d} [{shot['label']}] ---")
        results.append(process_shot(shot))

    (OUT_ROOT / "bench.json").write_text(json.dumps(results, indent=2))
    print(f"\nSummary: {OUT_ROOT / 'bench.json'}")
    total_frames = sum(r["n_out"] for r in results)
    total_wall = sum(r["wall_s"] for r in results)
    if total_wall > 0:
        print(f"  total: {total_frames} frames in {total_wall:.1f}s ({total_frames/total_wall:.1f} fps)")
    print(f"\nNext: .venv/bin/python colab/s06_eval/wire_to_viewer.py")
    print(f"Then: .venv/bin/python -m viewer.server  -> navigate to canonical-base, pick s06_vapoursynth_eval vs s05_damage_classify")


if __name__ == "__main__":
    main()
