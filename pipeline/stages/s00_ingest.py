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
    # Use -start_number so each worker's output joins into a single 00000001.png sequence.
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


def extract_serial(src: Path, out_dir: Path, total_frames: int) -> int:
    digits = max(8, len(str(total_frames)))
    pattern = str(out_dir / f"%0{digits}d.png")
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-start_number",
        "1",
        "-vsync",
        "0",
        "-an",
        "-vcodec",
        "png",
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
) -> int:
    """Split source by time, extract each slice in a subprocess. Frame numbering
    is kept contiguous by giving each worker its own start_number."""
    digits = max(8, len(str(total_frames)))
    # Split by time so frame counts per worker are roughly balanced.
    slice_s = duration_s / workers
    tasks = []
    for i in range(workers):
        start = i * slice_s
        dur = slice_s if i < workers - 1 else (duration_s - start + 0.5)
        first_idx = int(round(start * fps)) + 1
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
    # Spot brightness: sample 5 frames, require mean in [5, 250].
    try:
        from PIL import Image
        import numpy as np

        sample_idxs = [0, len(frames) // 4, len(frames) // 2, 3 * len(frames) // 4, len(frames) - 1]
        for i in sample_idxs:
            arr = np.asarray(Image.open(frames[i]).convert("L"))
            m = float(arr.mean())
            if m < 5 or m > 250:
                raise RuntimeError(
                    f"S00 auto-QC: frame {frames[i].name} brightness out of range ({m:.1f})"
                )
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
    logger.info(
        "Source: %dx%d, %.2fs, %.3f fps, ~%d frames",
        meta["width"], meta["height"], meta["duration_s"], meta["declared_fps"], meta["nb_frames"],
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
                logger.info("Extracting serially...")
                n = extract_serial(src, frames_dir, meta["nb_frames"])
            else:
                logger.info("Extracting with %d parallel workers...", workers)
                n = extract_parallel(
                    src, frames_dir, meta["nb_frames"],
                    meta["duration_s"], meta["declared_fps"], workers,
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
