"""Compare S02 variants on the current fixture run.

Produces, in the given run_dir:
- s02a_default/        — skimage phase_corr, smoothing=25, upsample=10 (already exists, skipped if present)
- s02b_wide/           — skimage phase_corr, smoothing=75, upsample=10 (aggressive: wider window)
- s02c_fine/           — skimage phase_corr, smoothing=25, upsample=50 (aggressive: finer sub-pixel)
- s02d_ffmpeg/         — ffmpeg `deshake` filter (block-matching, different family)

Each writes its own shake_metric.json. Script prints a summary comparison table.

Usage: python -m scripts.compare_s02_variants --run-dir runs/<timestamp>
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import click
import cv2
import numpy as np
from skimage.registration import phase_cross_correlation

from pipeline.common.logging import get_logger
from pipeline.common.paths import runs_dir
from pipeline.stages.s02_stabilise import (
    frame_to_frame_translation_rms,
    stabilise_shot_phase_corr,
)


def _newest_run() -> Path:
    candidates = sorted(runs_dir().iterdir(), reverse=True)
    for c in candidates:
        if any(d.name.startswith("s01_") for d in c.iterdir()):
            return c
    raise RuntimeError("No run with s01 output found.")


def _find_s00(run: Path) -> Path:
    return next(d for d in run.iterdir() if d.name.startswith("s00_"))


def _find_s01(run: Path) -> Path:
    return next(d for d in run.iterdir() if d.name.startswith("s01_"))


def _rename_existing_s02(run: Path):
    """If run already has s02_stabilise/, rename to s02a_default/ for clarity."""
    old = run / "s02_stabilise"
    new = run / "s02a_default"
    if old.exists() and not new.exists():
        old.rename(new)
        # also rename the frames subdir label for consistency
        # frames_stabilised is fine — stays the same
        return True
    return False


def _run_phase_corr_variant(
    name: str, run: Path, smoothing: int, upsample: int, logger,
) -> dict:
    s00 = _find_s00(run)
    s01 = _find_s01(run)
    frames = sorted((s00 / "frames_raw").glob("*.png"))
    probe = json.loads((s01 / "probe_report.json").read_text())
    intertitles = set(probe.get("intertitle_frames", []))
    shot_boundaries = probe.get("shot_boundaries", [])
    boundaries = sorted(set(shot_boundaries) | {0, len(frames)})
    ranges = list(zip(boundaries, boundaries[1:]))

    out_dir = run / name
    out_frames = out_dir / "frames_stabilised"
    if out_frames.exists():
        shutil.rmtree(out_frames)
    out_frames.mkdir(parents=True)

    logger.info("[%s] smoothing=%d upsample=%d ranges=%d", name, smoothing, upsample, len(ranges))
    t0 = time.perf_counter()
    for ri, (a, bnd) in enumerate(ranges):
        shot_paths = frames[a:bnd]
        stabilise_shot_phase_corr(
            shot_paths, intertitles, a, out_frames, smoothing, upsample, logger,
        )
    wall = time.perf_counter() - t0

    pre = frame_to_frame_translation_rms(frames)
    stab_paths = sorted(out_frames.glob("*.png"))
    post = frame_to_frame_translation_rms(stab_paths)
    result = {
        "variant": name,
        "method": "skimage_phase_corr",
        "smoothing": smoothing,
        "upsample": upsample,
        "pre_shake_rms": pre,
        "post_shake_rms": post,
        "reduction_pct": (1 - post / pre) * 100 if pre > 0 else 0.0,
        "wall_time_s": wall,
        "frames": len(stab_paths),
    }
    (out_dir / "shake_metric.json").write_text(json.dumps(result, indent=2))
    return result


def _run_median_anchor_second_pass(run: Path, window: int, logger) -> dict:
    """Second pass: align each s02a frame against the temporal median of a
    symmetric window of surrounding s02a frames. Orthogonal error mode:
    phase_corr chains pairwise (can accumulate drift); median-anchor anchors
    to a locally-stable reference (kills residual drift).
    """
    s00 = _find_s00(run)
    s01 = _find_s01(run)
    s02a_frames = sorted((run / "s02a_default" / "frames_stabilised").glob("*.png"))
    if not s02a_frames:
        raise RuntimeError("s02a_default output missing — run phase_corr first")
    probe = json.loads((s01 / "probe_report.json").read_text())
    intertitles = set(probe.get("intertitle_frames", []))
    shot_boundaries = probe.get("shot_boundaries", [])
    # Build per-frame → shot-range map so the median window doesn't cross shot cuts.
    boundaries = sorted(set(shot_boundaries) | {0, len(s02a_frames)})
    ranges = list(zip(boundaries, boundaries[1:]))
    frame_to_range = {}
    for a, b in ranges:
        for i in range(a, b):
            frame_to_range[i] = (a, b)

    out_dir = run / "s02e_median"
    out_frames = out_dir / "frames_stabilised"
    if out_frames.exists():
        shutil.rmtree(out_frames)
    out_frames.mkdir(parents=True)

    logger.info("[s02e_median] window=%d over %d s02a frames, %d shot ranges",
                window, len(s02a_frames), len(ranges))
    t0 = time.perf_counter()

    # Cache grayscale frames in memory for repeated median computations.
    grays: list[np.ndarray] = []
    for p in s02a_frames:
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if g is None:
            raise RuntimeError(f"Unreadable: {p}")
        grays.append(g)

    half = window // 2
    for i, p in enumerate(s02a_frames):
        if i in intertitles:
            shutil.copy2(p, out_frames / p.name)
            continue
        a, b = frame_to_range[i]
        # Clamp median window to shot boundaries to avoid cross-cut pollution.
        lo = max(a, i - half)
        hi = min(b, i + half + 1)
        if hi - lo < 3:
            # Not enough neighbours for a meaningful median; just copy.
            shutil.copy2(p, out_frames / p.name)
            continue
        window_stack = np.stack(grays[lo:hi], axis=0)
        median_frame = np.median(window_stack, axis=0).astype(np.float32)
        cur = grays[i].astype(np.float32)
        shift, _, _ = phase_cross_correlation(
            median_frame, cur, upsample_factor=10, normalization=None,
        )
        dy, dx = float(shift[0]), float(shift[1])
        color = cv2.imread(str(p))
        h, w = color.shape[:2]
        m = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float64)
        warped = cv2.warpAffine(
            color, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
        )
        cv2.imwrite(str(out_frames / p.name), warped)
        if (i + 1) % 50 == 0:
            logger.info("[s02e_median] %d/%d frames aligned to local median", i + 1, len(s02a_frames))
    wall = time.perf_counter() - t0

    # Metric vs original S00 (apples-to-apples with other variants).
    raw_frames = sorted((s00 / "frames_raw").glob("*.png"))
    pre = frame_to_frame_translation_rms(raw_frames)
    stab_paths = sorted(out_frames.glob("*.png"))
    post = frame_to_frame_translation_rms(stab_paths)
    result = {
        "variant": "s02e_median",
        "method": "phase_corr + median_anchor (stacked)",
        "window": window,
        "pre_shake_rms": pre,
        "post_shake_rms": post,
        "reduction_pct": (1 - post / pre) * 100 if pre > 0 else 0.0,
        "wall_time_s": wall,
        "frames": len(stab_paths),
    }
    (out_dir / "shake_metric.json").write_text(json.dumps(result, indent=2))
    return result


def _run_ffmpeg_deshake(run: Path, logger) -> dict:
    s00 = _find_s00(run)
    frames_in = s00 / "frames_raw"
    frame_names = sorted(frames_in.glob("*.png"))
    if not frame_names:
        raise RuntimeError("no input frames")
    digits = len(frame_names[0].stem)

    out_dir = run / "s02d_ffmpeg"
    out_frames = out_dir / "frames_stabilised"
    if out_frames.exists():
        shutil.rmtree(out_frames)
    out_frames.mkdir(parents=True)

    # ffmpeg deshake: per-frame block-matching. Runs in one pass.
    input_pattern = str(frames_in / f"%0{digits}d.png")
    output_pattern = str(out_frames / f"%0{digits}d.png")
    logger.info("[s02d_ffmpeg] running ffmpeg -vf deshake on %d frames", len(frame_names))
    t0 = time.perf_counter()
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-start_number", "1",
        "-i", input_pattern,
        "-vf", "deshake",
        "-vsync", "0",
        "-start_number", "1",
        output_pattern,
    ]
    subprocess.run(cmd, check=True)
    wall = time.perf_counter() - t0

    pre = frame_to_frame_translation_rms(frame_names)
    stab_paths = sorted(out_frames.glob("*.png"))
    post = frame_to_frame_translation_rms(stab_paths)
    result = {
        "variant": "s02d_ffmpeg",
        "method": "ffmpeg_deshake",
        "pre_shake_rms": pre,
        "post_shake_rms": post,
        "reduction_pct": (1 - post / pre) * 100 if pre > 0 else 0.0,
        "wall_time_s": wall,
        "frames": len(stab_paths),
    }
    (out_dir / "shake_metric.json").write_text(json.dumps(result, indent=2))
    return result


def _print_summary(results: list[dict]):
    print()
    print(f"{'variant':<20} {'method':<22} {'pre':>7} {'post':>7} {'reduce':>8} {'wall_s':>8}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['variant']:<20} "
            f"{r.get('method',''):<22} "
            f"{r['pre_shake_rms']:>7.3f} "
            f"{r['post_shake_rms']:>7.3f} "
            f"{r['reduction_pct']:>7.1f}% "
            f"{r['wall_time_s']:>7.1f}s"
        )


@click.command()
@click.option("--run-dir", "run_dir", default=None, help="Run dir to operate on (default: newest with s01).")
def main(run_dir: str | None):
    run = Path(run_dir) if run_dir else _newest_run()
    logger = get_logger("s02_compare", run)

    renamed = _rename_existing_s02(run)
    if renamed:
        logger.info("Renamed existing s02_stabilise → s02a_default for comparison clarity")

    # Recompute baseline metric for the existing default variant (for consistency with new variants).
    results: list[dict] = []
    s02a = run / "s02a_default"
    if s02a.exists():
        s00 = _find_s00(run)
        frames = sorted((s00 / "frames_raw").glob("*.png"))
        stab = sorted((s02a / "frames_stabilised").glob("*.png"))
        pre = frame_to_frame_translation_rms(frames)
        post = frame_to_frame_translation_rms(stab)
        results.append({
            "variant": "s02a_default",
            "method": "skimage_phase_corr",
            "smoothing": 25,
            "upsample": 10,
            "pre_shake_rms": pre,
            "post_shake_rms": post,
            "reduction_pct": (1 - post / pre) * 100 if pre > 0 else 0.0,
            "wall_time_s": 0.0,  # was run previously; wall-time not applicable
            "frames": len(stab),
        })

    results.append(_run_phase_corr_variant("s02b_wide", run, smoothing=75, upsample=10, logger=logger))
    results.append(_run_phase_corr_variant("s02c_fine", run, smoothing=25, upsample=50, logger=logger))
    results.append(_run_ffmpeg_deshake(run, logger))

    (run / "s02_variants_comparison.json").write_text(json.dumps(results, indent=2))
    _print_summary(results)


if __name__ == "__main__":
    main()
