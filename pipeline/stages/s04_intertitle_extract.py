"""S04 — Extract intertitle cards; produce an intertitle-free movie stream.

Reads S03's deflickered frames. Splits the stream into:
- `frames_movie/` — renumbered, contiguous PNGs with all intertitle frames removed
- `intertitles/card_NN/` — preserved card frames + metadata for OCR / re-typesetting
- `intertitle_plan.json` — reinsertion manifest for the final encode stage

Cards come from a user-verified list in config (not the noisy EAST auto-detect).
S05+ stages consume `frames_movie/` and a shot-boundary list remapped to the
new contiguous index space. Reinsertion happens at encode time (brief §16).

Empty `cards` list → no-op passthrough (frames copied through unchanged).
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path


def _link_or_copy(src: Path, dst: Path) -> None:
    """Prefer hardlink (zero disk cost; survives source deletion on the same FS).
    Falls back to copy if hardlinking fails (cross-filesystem, etc.).
    Downstream stages write new files — they never modify these, so sharing
    inodes with S03 is safe.
    """
    try:
        os.link(src, dst)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)

import click
import numpy as np

from pipeline.common.bench import bench_run
from pipeline.common.config import IntertitleCardCfg, load_config
from pipeline.common.hashing import sha256_file
from pipeline.common.logging import get_logger
from pipeline.common.paths import stage_dir, stage_input_frames

STAGE_ID = "s04"
STAGE_NAME = "intertitle_extract"


def find_stage(run_dir: Path, prefix: str) -> Path:
    for d in run_dir.iterdir():
        if d.is_dir() and d.name.startswith(prefix):
            return d
    raise FileNotFoundError(f"No {prefix}* under {run_dir}")


def load_probe(run_dir: Path) -> dict:
    return json.loads((find_stage(run_dir, "s01_") / "probe_report.json").read_text())


def build_frame_index_map(total_frames: int, cards: list[IntertitleCardCfg]) -> tuple[list[int], dict[int, int]]:
    """Return (new_to_orig, orig_to_new) mapping arrays.

    new_to_orig[k] = the original 0-indexed frame that becomes the k-th frame
                     in frames_movie/ (0-indexed).
    orig_to_new[i] = the new position of original frame i, or -1 if i lives
                     inside an extracted card (removed from the movie stream).
    """
    intertitle_indices: set[int] = set()
    for c in cards:
        for f in range(c.orig_start_frame, c.orig_end_frame + 1):
            intertitle_indices.add(f)

    new_to_orig = [i for i in range(total_frames) if i not in intertitle_indices]
    orig_to_new: dict[int, int] = {orig: new for new, orig in enumerate(new_to_orig)}
    return new_to_orig, orig_to_new


def remap_shot_boundaries(
    orig_boundaries: list[int],
    cards: list[IntertitleCardCfg],
    orig_to_new: dict[int, int],
    total_new: int,
) -> list[int]:
    """Map original shot-boundary indices into the new (movie-only) stream.

    - A boundary inside a card's range is dropped (no cut exists in the new stream there).
    - Other boundaries shift down by the count of intertitle frames preceding them.
    - Deduplicated and clamped to [0, total_new].
    """
    card_ranges = [(c.orig_start_frame, c.orig_end_frame) for c in cards]

    def inside_card(idx: int) -> bool:
        return any(a <= idx <= b for a, b in card_ranges)

    new_b: set[int] = set()
    for b in orig_boundaries:
        if inside_card(b):
            continue
        nb = orig_to_new.get(b)
        if nb is None:
            continue
        if 0 <= nb <= total_new:
            new_b.add(nb)
    # Always include the natural 0 start boundary if probe originally did.
    return sorted(new_b)


def build_intertitle_plan(
    cards: list[IntertitleCardCfg],
    fps: float,
    source_start_time_s: float,
    orig_to_new: dict[int, int],
    total_new: int,
) -> dict:
    """Reinsertion manifest — drives brief §16 intertitle_replace at encode time."""
    plan_cards = []
    for c in cards:
        # Insert position = new-stream index of the first frame that would
        # follow the card in the original stream. If the card is at the end,
        # insert at the tail (total_new).
        after_card_orig = c.orig_end_frame + 1
        while after_card_orig not in orig_to_new and after_card_orig < max(orig_to_new.keys(), default=-1) + 1:
            after_card_orig += 1
        new_insert_position = orig_to_new.get(after_card_orig, total_new)
        dur_s = (c.orig_end_frame - c.orig_start_frame + 1) / fps
        trimmed_start_s = c.orig_start_frame / fps
        plan_cards.append({
            "id": c.id,
            "label": c.label,
            "orig_start_frame": c.orig_start_frame,
            "orig_end_frame": c.orig_end_frame,
            "new_insert_position": new_insert_position,
            "duration_s": dur_s,
            "trimmed_time_s": round(trimmed_start_s, 3),
            "original_time_s": round(trimmed_start_s + source_start_time_s, 3),
            "regenerated_path": None,
        })
    total_runtime_s = (total_new + sum(c.orig_end_frame - c.orig_start_frame + 1 for c in cards)) / fps
    return {
        "source_fps": fps,
        "source_start_time_s": source_start_time_s,
        "cards": plan_cards,
        "final_runtime_with_intertitles_s": round(total_runtime_s, 3),
    }


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
            (c for c in candidates if any(d.name.startswith("s03_") for d in c.iterdir())),
            None,
        )
        if run_path is None:
            raise RuntimeError("No run with s03 output found. Run s03 first.")

    logger = get_logger("s04", run_path)
    logger.info("S04 intertitle_extract starting; run_dir=%s", run_path)

    in_frames_dir = stage_input_frames(run_path, "s04")
    frames = sorted(in_frames_dir.glob("*.png"))
    if not frames:
        raise RuntimeError(f"No frames in {in_frames_dir}")
    total = len(frames)

    probe = load_probe(run_path)
    orig_shot_boundaries: list[int] = probe.get("shot_boundaries", [])

    s04cfg = cfg.s04_intertitle_extract
    cards = list(s04cfg.cards)
    logger.info("Input: %d frames; cards to remove: %d", total, len(cards))
    for c in cards:
        n = c.orig_end_frame - c.orig_start_frame + 1
        logger.info("  %s: frames %d-%d (%d frames)", c.id, c.orig_start_frame, c.orig_end_frame, n)

    # Validate ranges.
    for c in cards:
        if not (0 <= c.orig_start_frame <= c.orig_end_frame < total):
            raise ValueError(f"Card {c.id} range out of bounds: {c.orig_start_frame}-{c.orig_end_frame}")

    out_dir = stage_dir(run_path, STAGE_ID, STAGE_NAME)
    frames_movie = out_dir / "frames_movie"
    intertitles_dir = out_dir / "intertitles"
    for d in (frames_movie, intertitles_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    new_to_orig, orig_to_new = build_frame_index_map(total, cards)
    logger.info("Movie stream: %d frames (removed %d intertitle frames)",
                len(new_to_orig), total - len(new_to_orig))

    with bench_run(f"{STAGE_ID}_{STAGE_NAME}", cfg.section_hash("s04"), out_dir) as b:
        # Write the movie stream (renumbered). Copy (not symlink) — downstream
        # stages need stable, independent files.
        digits = max(8, len(str(len(new_to_orig))))
        for new_idx, orig_idx in enumerate(new_to_orig):
            src = frames[orig_idx]
            dst = frames_movie / f"{new_idx + 1:0{digits}d}.png"
            _link_or_copy(src, dst)
        # Extract each card's frames + meta + representative.
        fps = probe.get("source_fps") or cfg.s00_ingest.parallel_workers  # fallback; real fps below
        # Prefer S00 metadata.json if present.
        try:
            s00_meta_path = find_stage(run_path, "s00_") / "metadata.json"
            fps = float(json.loads(s00_meta_path.read_text())["declared_fps"])
        except Exception:
            fps = 25.0
        for c in cards:
            card_dir = intertitles_dir / c.id.lower()
            frames_card = card_dir / "frames"
            frames_card.mkdir(parents=True)
            card_frames_orig = list(range(c.orig_start_frame, c.orig_end_frame + 1))
            for orig_idx in card_frames_orig:
                src = frames[orig_idx]
                _link_or_copy(src, frames_card / src.name)
            # Representative = middle frame (best single-frame OCR candidate).
            mid = card_frames_orig[len(card_frames_orig) // 2]
            _link_or_copy(frames[mid], card_dir / "representative.png")
            meta = {
                "id": c.id,
                "label": c.label,
                "orig_start_frame": c.orig_start_frame,
                "orig_end_frame": c.orig_end_frame,
                "frame_count": len(card_frames_orig),
                "duration_s": round(len(card_frames_orig) / fps, 3),
                "representative_orig_frame": mid,
            }
            (card_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        b.frames_processed = len(new_to_orig)

    # Shot boundaries remapped.
    new_boundaries = remap_shot_boundaries(
        orig_shot_boundaries, cards, orig_to_new, len(new_to_orig),
    )
    (out_dir / "shot_boundaries_adjusted.json").write_text(
        json.dumps({
            "source_shot_count": len(orig_shot_boundaries),
            "remapped_shot_count": len(new_boundaries),
            "shot_boundaries": new_boundaries,
        }, indent=2)
    )
    logger.info("Shot boundaries: %d → %d after card removal",
                len(orig_shot_boundaries), len(new_boundaries))

    # Frame index map (keep sparse JSON, not 7k entries).
    (out_dir / "frame_index_map.json").write_text(
        json.dumps({
            "total_original": total,
            "total_new": len(new_to_orig),
            "new_to_orig": new_to_orig,
        })
    )

    # Signal for S05+ that intertitles are gone.
    (out_dir / "intertitle_frames_after.json").write_text(json.dumps([]))

    # Reinsertion manifest.
    src_start = float(cfg.s00_ingest.source_start_time_s)
    plan = build_intertitle_plan(cards, fps, src_start, orig_to_new, len(new_to_orig))
    (out_dir / "intertitle_plan.json").write_text(json.dumps(plan, indent=2))

    # Auto-QC.
    movie_count = len(list(frames_movie.glob("*.png")))
    extracted_count = sum(c.orig_end_frame - c.orig_start_frame + 1 for c in cards)
    if movie_count + extracted_count != total:
        raise RuntimeError(
            f"S04 auto-QC: movie ({movie_count}) + extracted ({extracted_count}) "
            f"!= original ({total})"
        )
    # Byte-identity spot check: first frame of each card matches the original.
    for c in cards:
        orig = frames[c.orig_start_frame]
        card_copy = intertitles_dir / c.id.lower() / "frames" / orig.name
        if sha256_file(orig) != sha256_file(card_copy):
            raise RuntimeError(f"S04 auto-QC: card {c.id} first frame not byte-identical")
    logger.info("S04 auto-QC passed. Movie frames: %d  cards extracted: %d (%d frames)",
                movie_count, len(cards), extracted_count)


if __name__ == "__main__":
    main()
