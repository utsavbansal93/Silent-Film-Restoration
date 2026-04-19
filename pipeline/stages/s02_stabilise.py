"""S02 — Film weave removal (sub-pixel frame-to-frame registration).

The Phalke source is tripod-shot; what it actually has is **film weave**
(hand-crank gate jitter): sub-pixel frame-to-frame translation wobble from
uneven film transport. This is a registration problem, not a camera-shake
problem, so feature-tracking was the wrong tool (JOURNAL D11).

Algorithm (method=skimage_phase_corr):
  1. Per shot: compute sub-pixel shift between each consecutive pair of
     frames using FFT-based phase cross-correlation (robust to dirt/grain).
  2. Cumulate shifts into a trajectory (position of each frame relative to
     the shot's first frame).
  3. Smooth the trajectory with a wide centred moving average (default 25
     frames ≈ 1 s @ 25 fps) → the *intended* camera position at each frame.
  4. Subtract smoothed from raw trajectory → the *weave* component
     (high-frequency jitter).
  5. Warp each frame by -weave_component using cv2.warpAffine with sub-pixel
     translation. Pans and intentional motion are preserved; weave is removed.

Intertitle frames (per S01) are copied byte-identical — static cards don't
have weave to compute and shouldn't be warped.

shake_metric.json: phaseCorrelate-based RMS of per-pair translations, pre vs post.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import click
import cv2
import numpy as np
from skimage.registration import phase_cross_correlation

from pipeline.common.bench import bench_run
from pipeline.common.config import load_config
from pipeline.common.logging import get_logger
from pipeline.common.paths import stage_dir

STAGE_ID = "s02"
STAGE_NAME = "stabilise"


def find_stage(run_dir: Path, prefix: str) -> Path:
    for d in run_dir.iterdir():
        if d.is_dir() and d.name.startswith(prefix):
            return d
    raise FileNotFoundError(f"No {prefix}* under {run_dir}")


def load_probe(run_dir: Path) -> dict:
    s01 = find_stage(run_dir, "s01_")
    return json.loads((s01 / "probe_report.json").read_text())


def estimate_shift(prev_gray: np.ndarray, cur_gray: np.ndarray, upsample: int) -> tuple[float, float]:
    """Return (dy, dx) sub-pixel shift from prev → cur. FFT phase correlation."""
    shift, _error, _phasediff = phase_cross_correlation(
        prev_gray, cur_gray, upsample_factor=upsample, normalization=None,
    )
    return float(shift[0]), float(shift[1])


def smooth_1d(arr: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average. Edge-padded with reflection to preserve length."""
    if window <= 1 or len(arr) < 3:
        return arr.copy()
    w = min(window, len(arr))
    if w % 2 == 0:
        w += 1  # odd for true centering
    pad = w // 2
    padded = np.pad(arr, pad, mode="reflect")
    kernel = np.ones(w) / w
    return np.convolve(padded, kernel, mode="valid")


def stabilise_shot_phase_corr(
    frame_paths: list[Path],
    intertitle_set: set[int],
    global_offset: int,
    out_dir: Path,
    smoothing: int,
    upsample: int,
    logger,
) -> int:
    """Remove weave from one shot. Intertitle frames are copied through."""
    n = len(frame_paths)
    if n == 0:
        return 0
    if n == 1:
        shutil.copy2(frame_paths[0], out_dir / frame_paths[0].name)
        return 1

    # Read all grayscale frames once. At 1920×1080×1 byte = 2MB/frame; 800 frames = 1.6GB.
    # Fits comfortably on 16GB M3.
    grays = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in frame_paths]
    if any(g is None for g in grays):
        raise RuntimeError(f"Unreadable frame(s) in shot starting at index {global_offset}")

    # Per-pair sub-pixel shift. For intertitle-bordering pairs, treat as zero
    # (intertitles are static text cards; no weave between title↔motion frames).
    shifts = np.zeros((n - 1, 2), dtype=np.float64)  # columns: dy, dx
    for i in range(n - 1):
        gi = global_offset + i
        gi1 = global_offset + i + 1
        if gi in intertitle_set or gi1 in intertitle_set:
            continue
        dy, dx = estimate_shift(grays[i], grays[i + 1], upsample)
        shifts[i] = (dy, dx)
        if (i + 1) % 200 == 0:
            logger.info("phase_corr: %d/%d pairs in shot", i + 1, n - 1)

    # Integrate to trajectory relative to frame 0.
    trajectory = np.cumsum(shifts, axis=0)  # shape (n-1, 2)
    smoothed = np.stack([
        smooth_1d(trajectory[:, 0], smoothing),
        smooth_1d(trajectory[:, 1], smoothing),
    ], axis=1)
    # weave = trajectory - smoothed. Warping each frame by -weave cancels the jitter
    # while preserving the smoothed (intentional) motion.
    weave = trajectory - smoothed  # shape (n-1, 2)

    # Frame 0: unchanged (no preceding frame to register against).
    shutil.copy2(frame_paths[0], out_dir / frame_paths[0].name)
    count = 1

    # Frames 1..n-1: warp by -weave[i-1]. Intertitles pass-through.
    for i in range(1, n):
        g_idx = global_offset + i
        if g_idx in intertitle_set:
            shutil.copy2(frame_paths[i], out_dir / frame_paths[i].name)
            count += 1
            continue
        dy, dx = weave[i - 1]
        frame = cv2.imread(str(frame_paths[i]))
        h, w = frame.shape[:2]
        # skimage.phase_cross_correlation returns the shift required to register
        # `cur` onto `prev`. Cumulating gives the cumulative alignment correction;
        # weave = that correction minus its smooth (intended) component. To
        # cancel the weave on frame k we translate content by +weave (not -weave).
        m = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float64)
        warped = cv2.warpAffine(
            frame, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
        )
        cv2.imwrite(str(out_dir / frame_paths[i].name), warped)
        count += 1

    return count


def frame_to_frame_translation_rms(
    frame_paths: list[Path], upsample: int = 10, metric_size: tuple[int, int] = (640, 360),
) -> float:
    """Metric: RMS of per-pair sub-pixel translation magnitudes via skimage
    phase_cross_correlation. Frames are downsampled to `metric_size` before the
    FFT — ~9× faster than full-res with negligible precision loss since upsampling
    resolves the peak in a small neighbourhood regardless of source size. The
    reported shift is *in downsampled pixels*; pre and post are measured on the
    same grid so the comparison is apples-to-apples.
    """
    if len(frame_paths) < 2:
        return 0.0
    w, h = metric_size
    prev = cv2.imread(str(frame_paths[0]), cv2.IMREAD_GRAYSCALE)
    if prev is None:
        return 0.0
    prev = cv2.resize(prev, (w, h))
    mags = []
    for p in frame_paths[1:]:
        cur = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if cur is None:
            continue
        cur = cv2.resize(cur, (w, h))
        shift, _, _ = phase_cross_correlation(
            prev, cur, upsample_factor=upsample, normalization=None,
        )
        mags.append(float(shift[0]) ** 2 + float(shift[1]) ** 2)
        prev = cur
    if not mags:
        return 0.0
    return float(np.sqrt(np.mean(mags)))


@click.command()
@click.option("--config", "cfg_path", required=True)
@click.option("--run-dir", "run_dir", default=None)
def main(cfg_path: str, run_dir: str | None) -> None:
    cfg = load_config(cfg_path)

    if run_dir:
        run_path = Path(run_dir)
    else:
        from pipeline.common.paths import runs_dir
        candidates = sorted(runs_dir().iterdir(), reverse=True)
        run_path = next(
            (c for c in candidates if any(d.name.startswith("s01_") for d in c.iterdir())),
            None,
        )
        if run_path is None:
            raise RuntimeError("No run with s01 output found. Run s01 first.")

    logger = get_logger("s02", run_path)
    logger.info("S02 stabilise starting; run_dir=%s", run_path)

    s00 = find_stage(run_path, "s00_")
    in_frames_dir = s00 / "frames_raw"
    frames = sorted(in_frames_dir.glob("*.png"))
    if not frames:
        raise RuntimeError(f"No frames in {in_frames_dir}")

    probe = load_probe(run_path)
    shot_boundaries: list[int] = probe.get("shot_boundaries", [])
    intertitles: set[int] = set(probe.get("intertitle_frames", []))
    logger.info("Using %d shot boundaries, %d intertitle frames (pass-through)",
                len(shot_boundaries), len(intertitles))

    out_dir = stage_dir(run_path, STAGE_ID, STAGE_NAME)
    out_frames_dir = out_dir / "frames_stabilised"
    if out_frames_dir.exists():
        shutil.rmtree(out_frames_dir)
    out_frames_dir.mkdir()

    s02cfg = cfg.s02_stabilise
    logger.info("Method: %s", s02cfg.method)

    total_written = 0
    with bench_run(f"{STAGE_ID}_{STAGE_NAME}", cfg.section_hash("s02"), out_dir) as b:
        if s02cfg.method == "passthrough":
            logger.info("Passthrough: copying %d frames unchanged", len(frames))
            for p in frames:
                shutil.copy2(p, out_frames_dir / p.name)
            total_written = len(frames)
        elif s02cfg.method == "skimage_phase_corr":
            if s02cfg.per_shot and shot_boundaries:
                boundaries = sorted(set(shot_boundaries) | {0, len(frames)})
                ranges = list(zip(boundaries, boundaries[1:]))
            else:
                ranges = [(0, len(frames))]
            logger.info("Processing %d shot range(s); smoothing=%d, upsample=%d",
                        len(ranges), s02cfg.smoothing, s02cfg.upsample_factor)
            for ri, (a, bnd) in enumerate(ranges):
                shot_paths = frames[a:bnd]
                written = stabilise_shot_phase_corr(
                    shot_paths, intertitles, a, out_frames_dir,
                    s02cfg.smoothing, s02cfg.upsample_factor, logger,
                )
                total_written += written
                logger.info("Shot %d/%d: %d frames (range %d-%d)",
                            ri + 1, len(ranges), written, a, bnd - 1)
        else:
            # opencv_features deliberately removed from the default path.
            raise NotImplementedError(
                f"S02 method '{s02cfg.method}' not currently wired up. "
                "Use skimage_phase_corr (default) or passthrough."
            )
        b.frames_processed = total_written

    # Shake metric (same definition regardless of method — lets us compare).
    logger.info("Computing shake metric (pre)...")
    pre_rms = frame_to_frame_translation_rms(frames)
    stab_paths = sorted(out_frames_dir.glob("*.png"))
    logger.info("Computing shake metric (post)...")
    post_rms = frame_to_frame_translation_rms(stab_paths)

    shake = {
        "method": s02cfg.method,
        "pre_shake_rms": pre_rms,
        "post_shake_rms": post_rms,
        "reduction_pct": (1 - (post_rms / pre_rms)) * 100 if pre_rms > 0 else 0.0,
    }
    (out_dir / "shake_metric.json").write_text(json.dumps(shake, indent=2))
    logger.info("Shake RMS pre=%.4f post=%.4f (%.1f%% reduction)",
                pre_rms, post_rms, shake["reduction_pct"])

    # Auto-QC
    if len(stab_paths) != len(frames):
        raise RuntimeError(
            f"S02 auto-QC: frame count mismatch ({len(stab_paths)} vs {len(frames)})"
        )
    if s02cfg.method == "passthrough":
        logger.info("Passthrough mode: shake metric recorded for diagnostics only.")
    elif post_rms >= pre_rms:
        logger.warning(
            "S02 auto-QC (soft-fail): post-shake %.4f not less than pre-shake %.4f. "
            "Algorithm needs tuning (try wider smoothing window or higher upsample_factor).",
            post_rms, pre_rms,
        )
        shake["qc_soft_fail"] = True
        (out_dir / "shake_metric.json").write_text(json.dumps(shake, indent=2))
    else:
        logger.info("S02 auto-QC passed: shake reduced by %.1f%%", shake["reduction_pct"])

    # Intertitle byte-identity spot-check (non-passthrough only).
    if s02cfg.method != "passthrough" and s02cfg.skip_intertitles and intertitles:
        from pipeline.common.hashing import sha256_file
        for i in list(intertitles)[:10]:
            in_name = frames[i].name
            if sha256_file(frames[i]) != sha256_file(out_frames_dir / in_name):
                raise RuntimeError(f"S02 auto-QC: intertitle frame {in_name} was modified")


if __name__ == "__main__":
    main()
