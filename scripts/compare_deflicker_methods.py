"""Compare three deflicker approaches on a frame range of an existing run.

Methods:
  A. mean_norm       — per-frame luminance rescaled so each frame's mean matches
                       a rolling median-of-means over a window. Adjusts 1st moment.
  B. hist_match      — each frame's histogram matched to a rolling reference frame.
                       Adjusts the whole tonal distribution.
  C. ffmpeg_deflicker — ffmpeg -vf deflicker=size=5:mode=am (temporal avg).

Operates on S02 stabilised frames by default (isolates the deflicker contribution
from weave removal).

Writes outputs under <run>/s03_variants/{a_mean_norm,b_hist_match,c_ffmpeg}/
and a flicker_metric.json per variant. Prints a comparison table.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

import click
import cv2
import numpy as np
from skimage.exposure import match_histograms

from pipeline.common.logging import get_logger
from pipeline.common.paths import runs_dir

WINDOW = 25  # frames; rolling reference window


def flicker_rms(frame_paths: list[Path]) -> float:
    """RMS of frame-to-frame mean-luma differences. Independent of S02 metric."""
    if len(frame_paths) < 2:
        return 0.0
    prev_mean = None
    deltas = []
    for p in frame_paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        m = float(img.mean())
        if prev_mean is not None:
            deltas.append(m - prev_mean)
        prev_mean = m
    if not deltas:
        return 0.0
    return float(np.sqrt(np.mean(np.array(deltas) ** 2)))


def rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    """Centred rolling median over a 1-D array; mirrored padding."""
    w = min(window, len(values))
    if w % 2 == 0:
        w += 1
    pad = w // 2
    padded = np.pad(values, pad, mode="reflect")
    out = np.empty_like(values, dtype=np.float64)
    for i in range(len(values)):
        out[i] = np.median(padded[i:i + w])
    return out


def run_mean_norm(frames_in: list[Path], out_dir: Path, window: int, logger) -> int:
    logger.info("[A mean_norm] window=%d on %d frames", window, len(frames_in))
    # Compute per-frame mean.
    means = np.array([cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).mean() for p in frames_in])
    targets = rolling_median(means, window)
    # Rescale each frame's luminance by targets[i] / means[i].
    # Operate on colour frame's Y channel (BT.601) to preserve chroma relations.
    for i, p in enumerate(frames_in):
        img = cv2.imread(str(p))
        if img is None:
            shutil.copy2(p, out_dir / p.name)
            continue
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV).astype(np.float32)
        scale = targets[i] / max(means[i], 1e-6)
        yuv[..., 0] = np.clip(yuv[..., 0] * scale, 0, 255)
        rgb = cv2.cvtColor(yuv.astype(np.uint8), cv2.COLOR_YUV2BGR)
        cv2.imwrite(str(out_dir / p.name), rgb)
    return len(frames_in)


def run_hist_match(frames_in: list[Path], out_dir: Path, window: int, logger) -> int:
    logger.info("[B hist_match] window=%d on %d frames", window, len(frames_in))
    half = window // 2
    # Preload grayscale versions for median-frame computation (cheaper than full RGB).
    grays = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in frames_in]
    n = len(frames_in)
    for i, p in enumerate(frames_in):
        img = cv2.imread(str(p))
        if img is None:
            shutil.copy2(p, out_dir / p.name)
            continue
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        # Reference = pixel-wise median of grayscale window frames.
        ref_gray = np.median(np.stack(grays[lo:hi], axis=0), axis=0).astype(np.uint8)
        # Match the frame's L channel histogram to the reference's.
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L = lab[..., 0]
        L_matched = match_histograms(L, ref_gray).astype(np.uint8)
        lab[..., 0] = L_matched
        rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        cv2.imwrite(str(out_dir / p.name), rgb)
    return n


def run_ffmpeg_deflicker(frames_in: list[Path], out_dir: Path, logger) -> int:
    logger.info("[C ffmpeg_deflicker] on %d frames", len(frames_in))
    if not frames_in:
        return 0
    digits = len(frames_in[0].stem)
    in_dir = frames_in[0].parent
    start_num = int(frames_in[0].stem)
    n = len(frames_in)
    in_pattern = str(in_dir / f"%0{digits}d.png")
    out_pattern = str(out_dir / f"%0{digits}d.png")
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-start_number", str(start_num),
        "-i", in_pattern,
        "-frames:v", str(n),
        "-vf", "deflicker=size=5:mode=am",
        "-start_number", str(start_num),
        out_pattern,
    ]
    subprocess.run(cmd, check=True)
    return n


@click.command()
@click.option("--run-dir", "run_dir", default="runs/canonical-base")
@click.option("--start-frame", type=int, default=1579, help="Inclusive, 0-indexed into S02 output")
@click.option("--end-frame", type=int, default=1694, help="Inclusive, 0-indexed into S02 output")
@click.option("--window", type=int, default=WINDOW)
def main(run_dir: str, start_frame: int, end_frame: int, window: int):
    run = Path(run_dir)
    logger = get_logger("deflicker_compare", run)

    # Use S02 stabilised frames as input (isolates deflicker from weave).
    s02 = next(d for d in run.iterdir() if d.name.startswith("s02_"))
    all_frames = sorted((s02 / "frames_stabilised").glob("*.png"))
    subset = all_frames[start_frame:end_frame + 1]
    logger.info("Subset: frames %d-%d (%d frames, %.1fs)", start_frame, end_frame, len(subset), len(subset) / 25.0)

    variants_root = run / "s03_variants"
    variants_root.mkdir(exist_ok=True)

    results = []

    # Baseline: the S02 input itself (no deflicker).
    pre = flicker_rms(subset)
    results.append({"variant": "s02_input (baseline)", "flicker_rms": pre, "wall_s": 0.0, "reduction_pct": 0.0})

    for name, fn in [
        ("a_mean_norm", lambda d: run_mean_norm(subset, d, window, logger)),
        ("b_hist_match", lambda d: run_hist_match(subset, d, window, logger)),
        ("c_ffmpeg",    lambda d: run_ffmpeg_deflicker(subset, d, logger)),
    ]:
        out_dir = variants_root / name
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir()
        t0 = time.perf_counter()
        fn(out_dir)
        wall = time.perf_counter() - t0
        post_paths = sorted(out_dir.glob("*.png"))
        post = flicker_rms(post_paths)
        results.append({
            "variant": name,
            "flicker_rms": post,
            "wall_s": wall,
            "reduction_pct": (1 - post / pre) * 100 if pre > 0 else 0.0,
        })

    (variants_root / "flicker_comparison.json").write_text(json.dumps(results, indent=2))
    print()
    print(f"{'variant':<26} {'flicker_rms':>12} {'reduction':>10} {'wall_s':>8}")
    print("-" * 60)
    for r in results:
        print(f"{r['variant']:<26} {r['flicker_rms']:>12.4f} {r['reduction_pct']:>9.1f}% {r['wall_s']:>7.1f}s")


if __name__ == "__main__":
    main()
