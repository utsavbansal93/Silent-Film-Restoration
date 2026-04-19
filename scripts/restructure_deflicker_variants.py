"""Restructure deflicker comparison outputs into proper stage-like dirs
(s03a_mean_norm, s03b_hist_match, s03c_ffmpeg, s03bc_stacked) and add a
B→C stacked variant. Symlinks frames outside the test range to S02 so the
viewer can scrub across the full 7,821-frame sequence with only the test
window showing differences.
"""
from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

import click

from pipeline.common.logging import get_logger


def make_variant_dir(run: Path, name: str, variant_src: Path | None,
                     s02_frames_dir: Path, test_range: tuple[int, int],
                     logger) -> Path:
    """Create run/<name>/frames_deflickered/ as a full-sequence dir:
      - frames inside test_range are real PNGs from variant_src (or copied from s02 if None)
      - frames outside are relative symlinks to s02_frames_dir
    """
    out_dir = run / name / "frames_deflickered"
    if out_dir.exists():
        shutil.rmtree(out_dir.parent)
    out_dir.mkdir(parents=True)

    all_s02 = sorted(s02_frames_dir.glob("*.png"))
    start, end = test_range  # inclusive both ends, 0-indexed positions

    for i, p in enumerate(all_s02):
        target = out_dir / p.name
        if start <= i <= end and variant_src is not None:
            # Real PNG from the variant's output (filename matches).
            src = variant_src / p.name
            if src.exists():
                shutil.copy2(src, target)
            else:
                # Fallback: symlink to S02 if variant doesn't have that frame.
                rel = Path("..") / ".." / s02_frames_dir.relative_to(run) / p.name
                target.symlink_to(rel)
        else:
            # Symlink to S02 (relative path so viewer resolves it).
            rel = Path("..") / ".." / s02_frames_dir.relative_to(run) / p.name
            target.symlink_to(rel)

    real = sum(1 for p in out_dir.iterdir() if not p.is_symlink())
    links = sum(1 for p in out_dir.iterdir() if p.is_symlink())
    logger.info("[%s] real=%d symlinks=%d", name, real, links)
    return out_dir


def ffmpeg_deflicker_on_range(frames_in: list[Path], out_dir: Path, logger) -> None:
    """Run ffmpeg deflicker on a specific set of input frames, writing to out_dir."""
    if not frames_in:
        return
    digits = len(frames_in[0].stem)
    start_num = int(frames_in[0].stem)
    n = len(frames_in)
    in_pattern = str(frames_in[0].parent / f"%0{digits}d.png")
    out_pattern = str(out_dir / f"%0{digits}d.png")
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-start_number", str(start_num),
        "-i", in_pattern,
        "-frames:v", str(n),
        "-vf", "deflicker=size=5:mode=am",
        "-start_number", str(start_num),
        out_pattern,
    ]
    subprocess.run(cmd, check=True)


@click.command()
@click.option("--run-dir", "run_dir", default="runs/canonical-base")
@click.option("--start-frame", type=int, default=1579)
@click.option("--end-frame", type=int, default=1694)
def main(run_dir: str, start_frame: int, end_frame: int):
    run = Path(run_dir)
    logger = get_logger("restructure_deflicker", run)
    s02_frames = next(d for d in run.iterdir() if d.name.startswith("s02_")) / "frames_stabilised"
    assert s02_frames.exists()

    old_variants = run / "s03_variants"
    src_a = old_variants / "a_mean_norm" if (old_variants / "a_mean_norm").exists() else None
    src_b = old_variants / "b_hist_match" if (old_variants / "b_hist_match").exists() else None
    src_c = old_variants / "c_ffmpeg" if (old_variants / "c_ffmpeg").exists() else None

    if not (src_a and src_b and src_c):
        raise RuntimeError("Expected s03_variants/{a_mean_norm,b_hist_match,c_ffmpeg} from a prior run.")

    make_variant_dir(run, "s03a_mean_norm",  src_a, s02_frames, (start_frame, end_frame), logger)
    make_variant_dir(run, "s03b_hist_match", src_b, s02_frames, (start_frame, end_frame), logger)
    make_variant_dir(run, "s03c_ffmpeg",     src_c, s02_frames, (start_frame, end_frame), logger)

    # B→C stacked: apply ffmpeg deflicker on B's test-range real PNGs.
    logger.info("Building s03bc_stacked: ffmpeg deflicker on hist_match output")
    bc_dir = run / "s03bc_stacked" / "frames_deflickered"
    if bc_dir.exists():
        shutil.rmtree(bc_dir.parent)
    bc_dir.mkdir(parents=True)
    b_full = run / "s03b_hist_match" / "frames_deflickered"
    # Collect only real PNGs from B's test range as input.
    all_b = sorted(b_full.glob("*.png"))
    b_test = [all_b[i] for i in range(start_frame, end_frame + 1)]
    # ffmpeg will overwrite its output pattern; compute on test-range frames.
    # But we need a clean input that isn't a symlink — use B's real files directly.
    b_real_range = src_b
    t0 = time.perf_counter()
    ffmpeg_deflicker_on_range(sorted(b_real_range.glob("*.png")), bc_dir, logger)
    wall = time.perf_counter() - t0
    logger.info("B→C ffmpeg pass took %.1fs", wall)
    # Now symlink frames outside test range in bc_dir to S02.
    existing = {p.name for p in bc_dir.iterdir()}
    for i, p in enumerate(sorted(s02_frames.glob("*.png"))):
        if p.name not in existing:
            rel = Path("..") / ".." / s02_frames.relative_to(run) / p.name
            (bc_dir / p.name).symlink_to(rel)

    # Clean up the old s03_variants staging dir.
    shutil.rmtree(old_variants)
    logger.info("Removed old s03_variants/ staging dir.")
    logger.info("Variants ready: s03a_mean_norm, s03b_hist_match, s03c_ffmpeg, s03bc_stacked")


if __name__ == "__main__":
    main()
