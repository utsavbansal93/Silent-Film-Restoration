"""S02 — Stabilise using Python `vidstab` (OpenCV feature-tracking).

Per-shot mode: re-initialise the stabiliser at each shot boundary (from S01)
so smoothing doesn't drag across hard cuts.

Intertitle frames (from S01) are copied through unchanged — stabilising a
static card at best wastes work, at worst introduces wobble.

shake_metric.json:
- pre_shake_rms: RMS of frame-to-frame translation magnitudes on frames_raw/
- post_shake_rms: same on frames_stabilised/
- Auto-QC fails if post >= pre.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import click
import cv2
import numpy as np
from vidstab import VidStab

from pipeline.common.bench import bench_run
from pipeline.common.config import load_config
from pipeline.common.logging import get_logger
from pipeline.common.paths import new_run_dir, stage_dir

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


def frame_to_frame_translation_rms(frame_paths: list[Path]) -> float:
    """RMS of per-pair translation magnitudes using phaseCorrelate on grayscale frames.
    Cheap, reproducible, independent of the stabiliser's internals.
    """
    if len(frame_paths) < 2:
        return 0.0
    prev = cv2.imread(str(frame_paths[0]), cv2.IMREAD_GRAYSCALE)
    prev = cv2.resize(prev, (640, 360)).astype(np.float32)
    mags = []
    for p in frame_paths[1:]:
        cur = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if cur is None:
            continue
        cur = cv2.resize(cur, (640, 360)).astype(np.float32)
        (dx, dy), _ = cv2.phaseCorrelate(prev, cur)
        mags.append(dx * dx + dy * dy)
        prev = cur
    if not mags:
        return 0.0
    return float(np.sqrt(np.mean(mags)))


def stabilise_shot(
    in_paths: list[Path],
    out_dir: Path,
    smoothing: int,
) -> None:
    """Run VidStab on one shot's worth of frames. Writes outputs matching input names."""
    stabilizer = VidStab()
    # Prime the stabiliser by iterating through frames; it buffers internally.
    # VidStab's frame-by-frame API:
    #   stab_frame = stabilizer.stabilize_frame(input_frame=..., smoothing_window=smoothing)
    # Returns black padding for the first `smoothing` frames; we handle that.
    processed: list[tuple[Path, np.ndarray]] = []
    for p in in_paths:
        frame = cv2.imread(str(p))
        stab = stabilizer.stabilize_frame(input_frame=frame, smoothing_window=smoothing)
        if stab is None or stab.size == 0:
            # Warm-up: stabiliser is still buffering. Keep original for now.
            processed.append((p, frame))
        else:
            processed.append((p, stab))

    # Flush remaining buffered frames.
    tail: list[np.ndarray] = []
    while True:
        stab = stabilizer.stabilize_frame(input_frame=None, smoothing_window=smoothing)
        if stab is None or stab.size == 0:
            break
        tail.append(stab)
    # Replace the tail of `processed` with the flushed (true) stabilised frames.
    if tail:
        start = max(0, len(processed) - len(tail))
        for j, stab in enumerate(tail):
            if start + j < len(processed):
                processed[start + j] = (processed[start + j][0], stab)

    for p, frame in processed:
        cv2.imwrite(str(out_dir / p.name), frame)


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
    logger.info("Using %d shots, %d intertitle frames to pass through",
                len(shot_boundaries), len(intertitles))

    out_dir = stage_dir(run_path, STAGE_ID, STAGE_NAME)
    out_frames_dir = out_dir / "frames_stabilised"
    out_frames_dir.mkdir(exist_ok=True)

    s02cfg = cfg.s02_stabilise
    smoothing = s02cfg.smoothing

    # Build shot ranges. Shot boundaries are start-of-shot indices.
    if s02cfg.per_shot and shot_boundaries:
        ranges = []
        boundaries = sorted(set(shot_boundaries) | {0, len(frames)})
        for a, b in zip(boundaries, boundaries[1:]):
            ranges.append((a, b))
    else:
        ranges = [(0, len(frames))]
    logger.info("Processing %d shot ranges", len(ranges))

    with bench_run(f"{STAGE_ID}_{STAGE_NAME}", cfg.section_hash("s02"), out_dir) as b:
        b.frames_processed = len(frames)

        for ri, (a, bnd) in enumerate(ranges):
            shot_frames = frames[a:bnd]
            # Separate intertitles in this range from others.
            if s02cfg.skip_intertitles:
                non_title = [p for i, p in enumerate(shot_frames) if (a + i) not in intertitles]
                title = [p for i, p in enumerate(shot_frames) if (a + i) in intertitles]
                # Pass-through intertitles byte-for-byte.
                for p in title:
                    shutil.copy2(p, out_frames_dir / p.name)
            else:
                non_title = shot_frames
                title = []
            if non_title:
                stabilise_shot(non_title, out_frames_dir, smoothing)
            logger.info("Shot %d/%d: %d non-title + %d intertitle frames stabilised",
                        ri + 1, len(ranges), len(non_title), len(title))

    # shake_metric: pre vs post RMS of frame-to-frame translations.
    logger.info("Computing shake metric (pre)...")
    pre_rms = frame_to_frame_translation_rms(frames)
    stab_paths = sorted(out_frames_dir.glob("*.png"))
    logger.info("Computing shake metric (post)...")
    post_rms = frame_to_frame_translation_rms(stab_paths)

    shake = {
        "pre_shake_rms": pre_rms,
        "post_shake_rms": post_rms,
        "reduction_pct": (1 - (post_rms / pre_rms)) * 100 if pre_rms > 0 else 0.0,
    }
    (out_dir / "shake_metric.json").write_text(json.dumps(shake, indent=2))
    logger.info("Shake RMS pre=%.3f post=%.3f (%.1f%% reduction)",
                pre_rms, post_rms, shake["reduction_pct"])

    # Auto-QC
    if len(stab_paths) != len(frames):
        raise RuntimeError(
            f"S02 auto-QC: frame count mismatch ({len(stab_paths)} vs {len(frames)})"
        )
    if post_rms >= pre_rms:
        raise RuntimeError(
            f"S02 auto-QC: post-shake {post_rms:.3f} not less than pre-shake {pre_rms:.3f}"
        )
    # Intertitle frames must be byte-identical.
    if s02cfg.skip_intertitles and intertitles:
        from pipeline.common.hashing import sha256_file
        for i in list(intertitles)[:10]:  # check up to 10 for speed
            in_name = frames[i].name
            a = sha256_file(frames[i])
            b_h = sha256_file(out_frames_dir / in_name)
            if a != b_h:
                raise RuntimeError(
                    f"S02 auto-QC: intertitle frame {in_name} was modified"
                )
    logger.info("S02 auto-QC passed")


if __name__ == "__main__":
    main()
