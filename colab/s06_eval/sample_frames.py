"""Pick 50 stratified cat_a frames for S06 three-way model evaluation.

Stratification (10 each):
  dirty       highest dust/particulate score (localised bright specks)
  scratched   highest vertical-line score  (thin vertical Sobel-x ridges)
  splotched   highest local-patch anomaly  (32×32 patch max − frame mean)
  shot_edge   sampled evenly across cat_a frames flagged is_shot_boundary
  normal      near-median laplacian + brightness, not a shot boundary

S05's global stats don't separate scratches/splotches/dirt, so we compute
three local-damage scores here on 640×360 grayscale downsamples. ~1 min on
M3 with an 8-worker Pool.

Writes:
  runs/s06-eval/sample_frames.json   — {category: [{file, scores, ...}]}
  runs/s06-eval/input/<name>.png     — hardlinks to the 50 source frames
"""
from __future__ import annotations

import json
import os
import random
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
S05_DIR = ROOT / "runs" / "canonical-base" / "s05_damage_classify"
FRAMES_DIR = S05_DIR / "frames_classified"
OUT_DIR = ROOT / "runs" / "s06-eval"
INPUT_DIR = OUT_DIR / "input"

FEAT_W, FEAT_H = 640, 360
PATCH = 32
N_PER_CAT = 10

# User-forced scratched frames: 00001540.png (explicit) + 9 frames evenly
# sampled across shot 19 (frames 1450..1538, 3.56 s) which carries a long
# vertical scratch from top-right to bottom-right, per Utsav's review.
SCRATCH_SHOT_RANGE = (1450, 1538)
SCRATCH_EXTRA = ["00001540.png"]


def spatial_scores(path_str: str) -> tuple[str, float, float, float]:
    """Return (name, vertical_line_score, splotch_score, dust_score)."""
    img = cv2.imread(path_str, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return (Path(path_str).name, 0.0, 0.0, 0.0)
    img = cv2.resize(img, (FEAT_W, FEAT_H), interpolation=cv2.INTER_AREA)

    # Vertical scratches: Sobel-x magnitude summed per column, peak z-score.
    sx = np.abs(cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3))
    col_sum = sx.sum(axis=0)
    v_score = float((col_sum.max() - col_sum.mean()) / (col_sum.std() + 1e-6))

    # Splotches: max 32×32 patch mean minus frame mean.
    h, w = img.shape
    ph = h // PATCH
    pw = w // PATCH
    patches = img[: ph * PATCH, : pw * PATCH].reshape(ph, PATCH, pw, PATCH).mean(axis=(1, 3))
    s_score = float(patches.max() - img.mean())

    # Dust/particulate: fraction of pixels > mean + 3*std (specks on dark BG).
    thresh = img.mean() + 3 * img.std()
    d_score = float((img > thresh).mean())

    return (Path(path_str).name, v_score, s_score, d_score)


def main() -> None:
    random.seed(42)
    with (S05_DIR / "damage_map.json").open() as f:
        dmap = json.load(f)

    cat_a = {name: meta for name, meta in dmap["frames"].items() if meta["category"] == "cat_a"}
    print(f"cat_a frames: {len(cat_a)}")

    paths = [str(FRAMES_DIR / name) for name in cat_a]
    with Pool(os.cpu_count() or 4) as pool:
        results = pool.map(spatial_scores, paths, chunksize=32)

    scores = {name: (v, s, d) for name, v, s, d in results}

    def sort_by(idx: int):
        return sorted(scores.items(), key=lambda kv: kv[1][idx], reverse=True)

    chosen: dict[str, list[dict]] = {}
    used: set[str] = set()

    def pick_top(cat: str, sort_idx: int) -> None:
        bucket: list[dict] = []
        for name, (v, s, d) in sort_by(sort_idx):
            if name in used:
                continue
            bucket.append({
                "file": name,
                "v_score": round(v, 3),
                "splotch": round(s, 3),
                "dust": round(d, 4),
                "lap": cat_a[name]["scores"]["laplacian_var"],
                "brightness": cat_a[name]["scores"]["mean_brightness"],
                "is_shot_boundary": cat_a[name]["scores"]["is_shot_boundary"],
            })
            used.add(name)
            if len(bucket) == N_PER_CAT:
                break
        chosen[cat] = bucket

    # Scratched: user-forced. Shot 19 (1450..1538, 3.56s) carries a long
    # vertical scratch; include 00001540 explicitly and 9 evenly-spaced
    # frames from that shot.
    lo, hi = SCRATCH_SHOT_RANGE
    n_shot_samples = N_PER_CAT - len(SCRATCH_EXTRA)
    shot_samples = [
        f"{int(lo + i * (hi - lo) / max(1, n_shot_samples - 1)):08d}.png"
        for i in range(n_shot_samples)
    ]
    scratched_files = list(dict.fromkeys(SCRATCH_EXTRA + shot_samples))[:N_PER_CAT]
    scratched: list[dict] = []
    for name in scratched_files:
        if name not in cat_a:
            raise RuntimeError(f"forced scratched frame {name} not in cat_a")
        used.add(name)
        v, s, d = scores[name]
        scratched.append({
            "file": name,
            "v_score": round(v, 3),
            "splotch": round(s, 3),
            "dust": round(d, 4),
            "lap": cat_a[name]["scores"]["laplacian_var"],
            "brightness": cat_a[name]["scores"]["mean_brightness"],
            "is_shot_boundary": cat_a[name]["scores"]["is_shot_boundary"],
        })
    chosen["scratched"] = scratched

    # Order matters: most-specific categories first so they claim their frames.
    pick_top("splotched", 1)  # splotch score
    pick_top("dirty", 2)      # dust score

    # Shot-edge: evenly sampled across cat_a shot-boundary frames, not yet used.
    sb = [n for n, m in cat_a.items() if m["scores"]["is_shot_boundary"] and n not in used]
    sb_sorted = sorted(sb)
    step = max(1, len(sb_sorted) // N_PER_CAT)
    shot_edge: list[dict] = []
    for name in sb_sorted[::step][:N_PER_CAT]:
        used.add(name)
        v, s, d = scores[name]
        shot_edge.append({
            "file": name,
            "v_score": round(v, 3),
            "splotch": round(s, 3),
            "dust": round(d, 4),
            "lap": cat_a[name]["scores"]["laplacian_var"],
            "brightness": cat_a[name]["scores"]["mean_brightness"],
            "is_shot_boundary": True,
        })
    chosen["shot_edge"] = shot_edge

    # Normal: closest to median laplacian AND brightness, not shot boundary, unused.
    laps = np.array([m["scores"]["laplacian_var"] for m in cat_a.values()])
    brts = np.array([m["scores"]["mean_brightness"] for m in cat_a.values()])
    lap_med, brt_med = float(np.median(laps)), float(np.median(brts))
    candidates = []
    for name, m in cat_a.items():
        if name in used or m["scores"]["is_shot_boundary"]:
            continue
        lap, brt = m["scores"]["laplacian_var"], m["scores"]["mean_brightness"]
        dist = ((lap - lap_med) / (lap_med + 1e-6)) ** 2 + ((brt - brt_med) / (brt_med + 1e-6)) ** 2
        candidates.append((dist, name))
    candidates.sort()
    normal: list[dict] = []
    for _, name in candidates[:N_PER_CAT]:
        used.add(name)
        v, s, d = scores[name]
        normal.append({
            "file": name,
            "v_score": round(v, 3),
            "splotch": round(s, 3),
            "dust": round(d, 4),
            "lap": cat_a[name]["scores"]["laplacian_var"],
            "brightness": cat_a[name]["scores"]["mean_brightness"],
            "is_shot_boundary": False,
        })
    chosen["normal"] = normal

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Clear any stale hardlinks from a previous run.
    for existing in INPUT_DIR.glob("*.png"):
        existing.unlink()

    for cat, items in chosen.items():
        for item in items:
            src = FRAMES_DIR / item["file"]
            dst = INPUT_DIR / item["file"]
            os.link(src, dst)

    with (OUT_DIR / "sample_frames.json").open("w") as f:
        json.dump({
            "strata": {cat: [it["file"] for it in items] for cat, items in chosen.items()},
            "details": chosen,
            "total": sum(len(v) for v in chosen.values()),
        }, f, indent=2)

    print("\nStratified 50-frame sample:")
    for cat, items in chosen.items():
        print(f"  {cat:10s}  {len(items)}  e.g. {items[0]['file']}  lap={items[0]['lap']:.1f}  brt={items[0]['brightness']:.1f}")
    print(f"\nInputs hardlinked to: {INPUT_DIR}")
    print(f"Manifest:             {OUT_DIR / 'sample_frames.json'}")


if __name__ == "__main__":
    main()
