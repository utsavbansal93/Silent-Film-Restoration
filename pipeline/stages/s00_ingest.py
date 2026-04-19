"""S00 — Ingest source video into a PNG sequence + metadata.json.

Idempotent: on rerun, skips frames already extracted.
Writes bench.json. Emits metadata.json with resolution, duration, declared fps,
source sha256, and frame count.

Parallel extraction: splits source into N time-ranges, extracts each in a
subprocess, concatenates frame numbering. Controlled by
s00_ingest.parallel_workers in config. On M3 Air with fast SSD, >2x speedup
is typical for 4 workers.
"""
from __future__ import annotations

import json
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import click
import ffmpeg

from pipeline.common.bench import bench_run
from pipeline.common.config import PipelineConfig, load_config
from pipeline.common.hashing import sha256_file
from pipeline.common.logging import get_logger
from pipeline.common.paths import new_run_dir, project_root, stage_dir

STAGE_ID = "s00"
STAGE_NAME = "ingest"


def probe_source(src: Path) -> dict:
    info = ffmpeg.probe(str(src))
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    # declared fps from avg_frame_rate (num/den)
    num, den = v["avg_frame_rate"].split("/")
    fps = float(num) / float(den) if float(den) else 0.0
    duration = float(info["format"].get("duration", 0.0))
    return {
        "width": int(v["width"]),
        "height": int(v["height"]),
        "duration_s": duration,
        "declared_fps": fps,
        "codec": v.get("codec_name"),
        "nb_frames": int(v.get("nb_frames", 0)) or int(round(duration * fps)),
    }


def _extract_range(args: tuple) -> tuple[int, int]:
    src, out_dir, start_s, duration_s, first_idx, total_digits = args
    # -ss *before* -i is fast-seek; accurate enough for PNG frame extraction
    # (individual frames are decoded exactly).
    pattern = str(Path(out_dir) / f"%0{total_digits}d.png")
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_s}",
        "-i",
        str(src),
        "-t",
        f"{duration_s}",
        "-start_number",
        str(first_idx),
        "-vsync",
        "0",
        "-an",
        "-vcodec",
        "png",
        pattern,
    ]
    subprocess.run(cmd, check=True)
    # Count frames produced by this worker.
    produced = sum(
        1
        for _ in Path(out_dir).glob(f"*.png")
    )
    return (first_idx, produced)


def extract_serial(
    src: Path, out_dir: Path, total_frames: int, source_start_s: float = 0.0,
) -> int:
    digits = max(8, len(str(total_frames)))
    pattern = str(out_dir / f"%0{digits}d.png")
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    if source_start_s > 0:
        cmd += ["-ss", f"{source_start_s}"]
    cmd += [
        "-i", str(src),
        "-start_number", "1",
        "-vsync", "0",
        "-an",
        "-vcodec", "png",
        pattern,
    ]
    subprocess.run(cmd, check=True)
    return len(list(out_dir.glob("*.png")))


def extract_parallel(
    src: Path,
    out_dir: Path,
    total_frames: int,
    duration_s: float,
    fps: float,
    workers: int,
    source_start_s: float = 0.0,
) -> int:
    """Split source by time, extract each slice in a subprocess. Frame numbering
    is kept contiguous by giving each worker its own start_number."""
    digits = max(8, len(str(total_frames)))
    effective_duration = max(0.0, duration_s - source_start_s)
    slice_s = effective_duration / workers
    tasks = []
    for i in range(workers):
        start = source_start_s + i * slice_s
        dur = slice_s if i < workers - 1 else (effective_duration - i * slice_s + 0.5)
        # Frame numbering restarts from 1 for the clipped output.
        first_idx = int(round((i * slice_s) * fps)) + 1
        tasks.append((str(src), str(out_dir), start, dur, first_idx, digits))

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_extract_range, t) for t in tasks]
        for fut in as_completed(futures):
            fut.result()  # propagate errors

    return len(list(out_dir.glob("*.png")))


def auto_qc(frames_dir: Path, metadata: dict, logger) -> None:
    """Fail loudly on silent-failure modes."""
    frames = sorted(frames_dir.glob("*.png"))
    if not frames:
        raise RuntimeError("S00 auto-QC: no frames extracted")
    if metadata.get("nb_frames"):
        expected = metadata["nb_frames"]
        # Tolerate +-1 frame from ffmpeg rounding.
        if abs(len(frames) - expected) > max(2, int(expected * 0.005)):
            raise RuntimeError(
                f"S00 auto-QC: frame count {len(frames)} diverges from declared {expected}"
            )
    # Spot brightness: sample 5 frames. Silent films legitimately have near-black
    # leader/dark frames, so only fail if *all* samples are degenerate (catches
    # silent-failure modes like uniformly black output, not a legit dark leader).
    try:
        from PIL import Image
        import numpy as np

        sample_idxs = [0, len(frames) // 4, len(frames) // 2, 3 * len(frames) // 4, len(frames) - 1]
        means = []
        for i in sample_idxs:
            arr = np.asarray(Image.open(frames[i]).convert("L"))
            means.append(float(arr.mean()))
        if all(m < 5 for m in means):
            raise RuntimeError(
                f"S00 auto-QC: all sampled frames near-black (means={means}) — likely silent failure"
            )
        if all(m > 250 for m in means):
            raise RuntimeError(
                f"S00 auto-QC: all sampled frames blown out (means={means}) — likely silent failure"
            )
        logger.info("Brightness sample means: %s", [round(m, 1) for m in means])
    except ImportError:
        logger.warning("PIL/numpy not available; skipping brightness QC")


@click.command()
@click.option("--config", "cfg_path", required=True, help="Path to YAML config")
@click.option("--run-dir", "run_dir", default=None, help="Override run directory")
def main(cfg_path: str, run_dir: str | None) -> None:
    cfg = load_config(cfg_path)
    run_path = Path(run_dir) if run_dir else new_run_dir(cfg.profile_name)
    logger = get_logger("s00", run_path)
    logger.info("S00 ingest starting; run_dir=%s", run_path)

    # Resolve source path (relative to project root if not absolute).
    src = Path(cfg.source.path)
    if not src.is_absolute():
        src = project_root() / src
    if not src.exists():
        raise FileNotFoundError(f"Source not found: {src}")

    out_dir = stage_dir(run_path, STAGE_ID, STAGE_NAME)
    frames_dir = out_dir / "frames_raw"
    frames_dir.mkdir(exist_ok=True)

    # Hash source once (cached).
    src_hash = sha256_file(src)
    logger.info("Source sha256: %s", src_hash)

    meta = probe_source(src)
    meta["source_path"] = str(src.relative_to(project_root()))
    meta["source_sha256"] = src_hash

    source_start_s = cfg.s00_ingest.source_start_time_s
    if source_start_s > 0:
        effective_duration = max(0.0, meta["duration_s"] - source_start_s)
        effective_frames = int(round(effective_duration * meta["declared_fps"]))
        meta["source_start_time_s"] = source_start_s
        meta["effective_duration_s"] = effective_duration
        meta["nb_frames_original"] = meta["nb_frames"]
        meta["nb_frames"] = effective_frames
        logger.info(
            "Source: %dx%d, %.2fs (offset %.3fs → %.2fs effective, %d frames), %.3f fps",
            meta["width"], meta["height"], meta["duration_s"],
            source_start_s, effective_duration, effective_frames, meta["declared_fps"],
        )
    else:
        logger.info(
            "Source: %dx%d, %.2fs, %.3f fps, ~%d frames",
            meta["width"], meta["height"], meta["duration_s"],
            meta["declared_fps"], meta["nb_frames"],
        )

    with bench_run(f"{STAGE_ID}_{STAGE_NAME}", cfg.section_hash("s00"), out_dir) as b:
        b.input_hash = src_hash
        workers = cfg.s00_ingest.parallel_workers
        # If frames already present (idempotent), skip.
        existing = list(frames_dir.glob("*.png"))
        if len(existing) >= meta["nb_frames"] - 2:
            logger.info("Frames already extracted (%d); skipping", len(existing))
            n = len(existing)
        else:
            # Clear any partial state to avoid mixing start_numbers.
            for p in existing:
                p.unlink()
            if workers <= 1:
                logger.info("Extracting serially (offset=%.3fs)...", source_start_s)
                n = extract_serial(
                    src, frames_dir, meta["nb_frames"], source_start_s=source_start_s,
                )
            else:
                logger.info("Extracting with %d parallel workers (offset=%.3fs)...",
                            workers, source_start_s)
                n = extract_parallel(
                    src, frames_dir, meta["nb_frames"],
                    meta["duration_s"], meta["declared_fps"], workers,
                    source_start_s=source_start_s,
                )
        b.frames_processed = n
        meta["frames_extracted"] = n

    # Write metadata.json
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    logger.info("Wrote metadata.json; %d frames", n)

    auto_qc(frames_dir, meta, logger)
    logger.info("S00 auto-QC passed")


if __name__ == "__main__":
    main()
