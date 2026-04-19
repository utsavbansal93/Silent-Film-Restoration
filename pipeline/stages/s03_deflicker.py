"""S03 — Deflicker via per-shot rolling-histogram matching.

Silent-era nitrate film has frame-to-frame luminance wobble from hand-cranked
exposure + chemistry drift. Method `hist_match` matches each frame's luminance
distribution to a rolling reference (pixel-wise median of neighbouring frames
inside the same shot). Subsumes mean normalisation and gave 91% flicker
reduction on the 4.6 s sample vs ffmpeg deflicker's 51% (JOURNAL D17).

Per-shot mode prevents reference-leak across hard cuts. Intertitle frames are
copied through byte-identical.

Reads:  S02's frames_stabilised/ (so deflicker sees the weave-corrected film)
Writes: S03's frames_deflickered/

Metric: RMS of frame-to-frame mean-luma differences (pre vs post).
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

from pipeline.common.bench import bench_run
from pipeline.common.config import load_config
from pipeline.common.logging import get_logger
from pipeline.common.paths import stage_dir

STAGE_ID = "s03"
STAGE_NAME = "deflicker"


def find_stage(run_dir: Path, prefix: str) -> Path:
    for d in run_dir.iterdir():
        if d.is_dir() and d.name.startswith(prefix):
            return d
    raise FileNotFoundError(f"No {prefix}* under {run_dir}")


def load_probe(run_dir: Path) -> dict:
    s01 = find_stage(run_dir, "s01_")
    return json.loads((s01 / "probe_report.json").read_text())


def frame_mean_luma(p: Path) -> float:
    g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return float(g.mean()) if g is not None else 0.0


def flicker_rms(frame_paths: list[Path]) -> float:
    """RMS of frame-to-frame mean-luma differences."""
    if len(frame_paths) < 2:
        return 0.0
    deltas = []
    prev = frame_mean_luma(frame_paths[0])
    for p in frame_paths[1:]:
        cur = frame_mean_luma(p)
        deltas.append(cur - prev)
        prev = cur
    if not deltas:
        return 0.0
    return float(np.sqrt(np.mean(np.array(deltas) ** 2)))


def hist_match_shot(
    frame_paths: list[Path],
    out_dir: Path,
    window: int,
    intertitle_set: set[int],
    global_offset: int,
    logger,
) -> int:
    """Match each non-intertitle frame's luminance to a rolling reference
    median computed from *grayscale* window frames inside this shot."""
    n = len(frame_paths)
    if n == 0:
        return 0
    if n == 1:
        shutil.copy2(frame_paths[0], out_dir / frame_paths[0].name)
        return 1

    # Preload grayscale for reference computation.
    grays = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in frame_paths]
    half = window // 2

    count = 0
    for i, p in enumerate(frame_paths):
        g_idx = global_offset + i
        if g_idx in intertitle_set:
            shutil.copy2(p, out_dir / p.name)
            count += 1
            continue
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        if hi - lo < 3:
            # Too few neighbours — just copy.
            shutil.copy2(p, out_dir / p.name)
            count += 1
            continue
        ref_gray = np.median(np.stack(grays[lo:hi], axis=0), axis=0).astype(np.uint8)
        img = cv2.imread(str(p))
        if img is None:
            shutil.copy2(p, out_dir / p.name)
            count += 1
            continue
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[..., 0] = match_histograms(lab[..., 0], ref_gray).astype(np.uint8)
        cv2.imwrite(str(out_dir / p.name), cv2.cvtColor(lab, cv2.COLOR_LAB2BGR))
        count += 1
        if (i + 1) % 500 == 0:
            logger.info("shot progress: %d/%d", i + 1, n)
    return count


def mean_norm_shot(
    frame_paths: list[Path],
    out_dir: Path,
    window: int,
    intertitle_set: set[int],
    global_offset: int,
    logger,
) -> int:
    n = len(frame_paths)
    if n == 0:
        return 0
    means = np.array([frame_mean_luma(p) for p in frame_paths])
    # Centred rolling median.
    w = min(window, n)
    if w % 2 == 0:
        w += 1
    pad = w // 2
    padded = np.pad(means, pad, mode="reflect")
    targets = np.array([np.median(padded[i:i + w]) for i in range(n)])

    count = 0
    for i, p in enumerate(frame_paths):
        g_idx = global_offset + i
        if g_idx in intertitle_set:
            shutil.copy2(p, out_dir / p.name)
            count += 1
            continue
        img = cv2.imread(str(p))
        if img is None:
            shutil.copy2(p, out_dir / p.name)
            count += 1
            continue
        scale = targets[i] / max(means[i], 1e-6)
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV).astype(np.float32)
        yuv[..., 0] = np.clip(yuv[..., 0] * scale, 0, 255)
        cv2.imwrite(str(out_dir / p.name), cv2.cvtColor(yuv.astype(np.uint8), cv2.COLOR_YUV2BGR))
        count += 1
    return count


def ffmpeg_deflicker_shot(
    frame_paths: list[Path], out_dir: Path, logger,
) -> int:
    if not frame_paths:
        return 0
    digits = len(frame_paths[0].stem)
    start = int(frame_paths[0].stem)
    n = len(frame_paths)
    in_pattern = str(frame_paths[0].parent / f"%0{digits}d.png")
    out_pattern = str(out_dir / f"%0{digits}d.png")
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-start_number", str(start),
        "-i", in_pattern,
        "-frames:v", str(n),
        "-vf", "deflicker=size=5:mode=am",
        "-start_number", str(start),
        out_pattern,
    ]
    subprocess.run(cmd, check=True)
    return n


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
            (c for c in candidates if any(d.name.startswith("s02_") for d in c.iterdir())),
            None,
        )
        if run_path is None:
            raise RuntimeError("No run with s02 output found. Run s02 first.")

    logger = get_logger("s03", run_path)
    logger.info("S03 deflicker starting; run_dir=%s", run_path)

    s02 = find_stage(run_path, "s02_")
    in_frames_dir = s02 / "frames_stabilised"
    frames = sorted(in_frames_dir.glob("*.png"))
    if not frames:
        raise RuntimeError(f"No frames in {in_frames_dir}")

    probe = load_probe(run_path)
    shot_boundaries: list[int] = probe.get("shot_boundaries", [])
    intertitles: set[int] = set(probe.get("intertitle_frames", []))
    logger.info("Using %d shot boundaries, %d intertitle frames (pass-through)",
                len(shot_boundaries), len(intertitles))

    out_dir = stage_dir(run_path, STAGE_ID, STAGE_NAME)
    out_frames_dir = out_dir / "frames_deflickered"
    if out_frames_dir.exists():
        shutil.rmtree(out_frames_dir)
    out_frames_dir.mkdir()

    s03cfg = cfg.s03_deflicker
    logger.info("Method: %s  window: %d  per_shot: %s  skip_intertitles: %s",
                s03cfg.method, s03cfg.window, s03cfg.per_shot, s03cfg.skip_intertitles)

    total_written = 0
    t_stage0 = time.perf_counter()
    with bench_run(f"{STAGE_ID}_{STAGE_NAME}", cfg.section_hash("s03"), out_dir) as b:
        if s03cfg.method == "passthrough":
            for p in frames:
                shutil.copy2(p, out_frames_dir / p.name)
            total_written = len(frames)
        else:
            if s03cfg.per_shot and shot_boundaries:
                boundaries = sorted(set(shot_boundaries) | {0, len(frames)})
                ranges = list(zip(boundaries, boundaries[1:]))
            else:
                ranges = [(0, len(frames))]
            logger.info("Processing %d shot range(s)", len(ranges))
            for ri, (a, bnd) in enumerate(ranges):
                shot_paths = frames[a:bnd]
                if s03cfg.method == "hist_match":
                    n_written = hist_match_shot(
                        shot_paths, out_frames_dir, s03cfg.window,
                        intertitles if s03cfg.skip_intertitles else set(),
                        a, logger,
                    )
                elif s03cfg.method == "mean_norm":
                    n_written = mean_norm_shot(
                        shot_paths, out_frames_dir, s03cfg.window,
                        intertitles if s03cfg.skip_intertitles else set(),
                        a, logger,
                    )
                elif s03cfg.method == "ffmpeg":
                    # ffmpeg doesn't natively skip intertitles; warn if asked.
                    if s03cfg.skip_intertitles and intertitles:
                        logger.warning("method=ffmpeg ignores skip_intertitles; intertitles will be processed")
                    n_written = ffmpeg_deflicker_shot(shot_paths, out_frames_dir, logger)
                else:
                    raise NotImplementedError(f"unknown method: {s03cfg.method}")
                total_written += n_written
                elapsed = time.perf_counter() - t_stage0
                pct = total_written / len(frames) * 100 if len(frames) else 0
                logger.info("Shot %d/%d: %d frames (range %d-%d) — %d/%d total (%.1f%%, %.0fs elapsed)",
                            ri + 1, len(ranges), n_written, a, bnd - 1,
                            total_written, len(frames), pct, elapsed)
        b.frames_processed = total_written

    # Metric
    logger.info("Computing flicker metric (pre)...")
    pre_rms = flicker_rms(frames)
    stab_paths = sorted(out_frames_dir.glob("*.png"))
    logger.info("Computing flicker metric (post)...")
    post_rms = flicker_rms(stab_paths)

    flicker = {
        "method": s03cfg.method,
        "pre_flicker_rms": pre_rms,
        "post_flicker_rms": post_rms,
        "reduction_pct": (1 - post_rms / pre_rms) * 100 if pre_rms > 0 else 0.0,
    }
    (out_dir / "flicker_metric.json").write_text(json.dumps(flicker, indent=2))
    logger.info("Flicker RMS pre=%.4f post=%.4f (%.1f%% reduction)",
                pre_rms, post_rms, flicker["reduction_pct"])

    # Auto-QC
    if len(stab_paths) != len(frames):
        raise RuntimeError(
            f"S03 auto-QC: frame count mismatch ({len(stab_paths)} vs {len(frames)})"
        )
    if s03cfg.method == "passthrough":
        logger.info("Passthrough mode: metric recorded for diagnostics only.")
    elif post_rms >= pre_rms:
        logger.warning(
            "S03 auto-QC (soft-fail): post-flicker %.4f not less than pre %.4f.",
            post_rms, pre_rms,
        )
        flicker["qc_soft_fail"] = True
        (out_dir / "flicker_metric.json").write_text(json.dumps(flicker, indent=2))
    else:
        logger.info("S03 auto-QC passed: flicker reduced by %.1f%%", flicker["reduction_pct"])

    # Intertitle byte-identity check (hist_match / mean_norm only; ffmpeg warns).
    if s03cfg.method in ("hist_match", "mean_norm") and s03cfg.skip_intertitles and intertitles:
        from pipeline.common.hashing import sha256_file
        for i in list(intertitles)[:10]:
            in_name = frames[i].name
            if sha256_file(frames[i]) != sha256_file(out_frames_dir / in_name):
                raise RuntimeError(f"S03 auto-QC: intertitle frame {in_name} was modified")


if __name__ == "__main__":
    main()
