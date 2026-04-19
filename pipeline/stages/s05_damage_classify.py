"""S05 — Damage classification: tag every movie-stream frame as cat_a / cat_b / cat_c.

Three categories (brief §3.4b):
  cat_a  structurally intact but dirty           → S06 dirt_remove handles
  cat_b  partially corrupted (torn/stained)      → S07 inpaint_partial handles
  cat_c  fully unusable                          → S08 drop_unusable removes

Two-pass algorithm:
  Pass 1 (parallel)    compute per-frame features via multiprocessing.Pool
  Pass 2 (sequential)  classify using percentile-derived thresholds +
                       temporal brightness signal

Laplacian thresholds adapt to this film's sharpness distribution by default
(percentile-based), avoiding absolute-value fragility on soft-focus 1917
footage. Absolute overrides are available in config.

Temporal signal: after Pass 1, a 5-frame centered rolling mean of brightness
is computed per-shot. A frame whose brightness deviates from the rolling mean
by more than `temporal_brightness_delta` is upgraded from cat_a → cat_b.
This catches single anomalous/over-exposed frames in otherwise stable scenes
without requiring full SSIM comparison.

Reads:   S04 frames_movie/               (6,890-frame intertitle-free stream)
         S04 shot_boundaries_adjusted.json (temporal windowing + boundary flags)
Writes:  damage_map.json                 per-frame classification + scores
         damage_summary.json             counts, percentages, QC flags
         threshold_calibration.json      per-feature distribution + resolved thresholds
         damage_gallery.html             visual review: up to 10 cat-b + 10 cat-c frames
         frames_classified/              hardlinks to frames_movie/ (zero disk; viewer)
         bench.json                      wall time, fps, peak memory
"""
from __future__ import annotations

import json
import os
import random
import shutil
import time
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import click
import cv2
import numpy as np

from pipeline.common.bench import bench_run
from pipeline.common.config import S05Cfg, load_config
from pipeline.common.logging import get_logger
from pipeline.common.paths import stage_dir, stage_input_frames

STAGE_ID = "s05"
STAGE_NAME = "damage_classify"

# Downsample target for feature computation.
# 640×360 resolves film-grain damage well; ~9× faster than full 1080p.
_FEAT_W, _FEAT_H = 640, 360


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FrameFeatures:
    laplacian_var: float        # sharpness proxy; low = blurry/flat/burned
    mean_brightness: float      # 0–255 mean pixel value
    dark_pixel_ratio: float     # fraction of pixels < 10
    bright_pixel_ratio: float   # fraction of pixels > 245


@dataclass
class ResolvedThresholds:
    cat_c_laplacian: float
    cat_b_laplacian: float
    cat_c_mean_low: float
    cat_c_mean_high: float
    cat_c_extreme_ratio: float
    cat_b_extreme_ratio: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_stage(run_dir: Path, prefix: str) -> Path:
    for d in run_dir.iterdir():
        if d.is_dir() and d.name.startswith(prefix):
            return d
    raise FileNotFoundError(f"No {prefix}* under {run_dir}")


def _link_or_copy(src: Path, dst: Path) -> None:
    """Prefer hardlink (zero disk if same FS). Falls back to copy."""
    try:
        os.link(src, dst)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# Pass 1: feature extraction (module-level for multiprocessing pickle)
# ---------------------------------------------------------------------------

def compute_features(img_path: Path) -> tuple[str, FrameFeatures]:
    """Pure function — grayscale-downsample and compute 4 heuristic features.

    Returns (filename, FrameFeatures). Module-level so Pool can pickle it.
    """
    gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        # Unreadable frame → maximally dark so it hits cat_c thresholds
        return img_path.name, FrameFeatures(
            laplacian_var=0.0, mean_brightness=0.0,
            dark_pixel_ratio=1.0, bright_pixel_ratio=0.0,
        )
    h, w = gray.shape
    if w > _FEAT_W:
        gray = cv2.resize(gray, (_FEAT_W, _FEAT_H), interpolation=cv2.INTER_AREA)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_b = float(gray.mean())
    dark = float((gray < 10).mean())
    bright = float((gray > 245).mean())
    return img_path.name, FrameFeatures(
        laplacian_var=lap_var,
        mean_brightness=mean_b,
        dark_pixel_ratio=dark,
        bright_pixel_ratio=bright,
    )


# ---------------------------------------------------------------------------
# Between passes: threshold resolution + rolling brightness
# ---------------------------------------------------------------------------

def resolve_thresholds(all_features: list[FrameFeatures], cfg: S05Cfg) -> ResolvedThresholds:
    """Derive classification thresholds from the feature distribution.

    Laplacian thresholds default to percentiles of this film's own sharpness
    distribution so soft-focus 1917 shots aren't mislabelled as unusable.
    Absolute overrides (cat_c_laplacian_abs / cat_b_laplacian_abs) bypass this.

    Brightness and extreme-pixel thresholds remain absolute — a frame that is
    essentially all-black or all-white is damaged regardless of scene context.
    """
    lap_vals = np.array([f.laplacian_var for f in all_features])
    cat_c_lap = (
        cfg.cat_c_laplacian_abs
        if cfg.cat_c_laplacian_abs is not None
        else float(np.percentile(lap_vals, cfg.cat_c_laplacian_pct))
    )
    cat_b_lap = (
        cfg.cat_b_laplacian_abs
        if cfg.cat_b_laplacian_abs is not None
        else float(np.percentile(lap_vals, cfg.cat_b_laplacian_pct))
    )
    # cat_b threshold must be >= cat_c (enforce ordering after overrides)
    cat_b_lap = max(cat_b_lap, cat_c_lap)
    return ResolvedThresholds(
        cat_c_laplacian=cat_c_lap,
        cat_b_laplacian=cat_b_lap,
        cat_c_mean_low=cfg.cat_c_mean_low,
        cat_c_mean_high=cfg.cat_c_mean_high,
        cat_c_extreme_ratio=cfg.cat_c_extreme_ratio,
        cat_b_extreme_ratio=cfg.cat_b_extreme_ratio,
    )


def compute_rolling_brightness(
    ordered_names: list[str],
    features_map: dict[str, FrameFeatures],
    shot_boundaries: list[int],
    window: int = 5,
) -> dict[str, float]:
    """5-frame centered rolling mean of mean_brightness, computed per-shot.

    Resetting at shot boundaries prevents a tonally-different card polluting
    the rolling mean of adjacent content shots.
    """
    total = len(ordered_names)
    # Build shot ranges as half-open intervals [a, b)
    shot_starts = sorted(set([0] + list(shot_boundaries)) | {total})
    shot_ranges = list(zip(shot_starts, shot_starts[1:]))
    half = window // 2
    result: dict[str, float] = {}
    for a, b in shot_ranges:
        shot_names = ordered_names[a:b]
        n = len(shot_names)
        if n == 0:
            continue
        brightnesses = np.array([features_map[name].mean_brightness for name in shot_names])
        for i, name in enumerate(shot_names):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            result[name] = float(brightnesses[lo:hi].mean())
    return result


# ---------------------------------------------------------------------------
# Pass 2: classification (pure function, unit-testable)
# ---------------------------------------------------------------------------

def classify_frame(
    features: FrameFeatures,
    thresholds: ResolvedThresholds,
    rolling_brightness_mean: float | None,
    cfg: S05Cfg,
) -> tuple[str, str | None]:
    """Pure function. Returns (category, reason_or_None).

    Evaluation order: cat_c checks → cat_b checks → temporal upgrade → cat_a.
    Reason string is human-readable; the caller prepends the filename for grep-ability.
    """
    extreme_ratio = features.dark_pixel_ratio + features.bright_pixel_ratio

    # --- cat_c: frame is essentially unusable ---
    if features.laplacian_var < thresholds.cat_c_laplacian:
        return "cat_c", (
            f"laplacian_var {features.laplacian_var:.1f} < cat_c threshold "
            f"{thresholds.cat_c_laplacian:.1f}"
        )
    if features.mean_brightness < thresholds.cat_c_mean_low:
        return "cat_c", (
            f"mean_brightness {features.mean_brightness:.1f} < {thresholds.cat_c_mean_low:.1f}"
        )
    if features.mean_brightness > thresholds.cat_c_mean_high:
        return "cat_c", (
            f"mean_brightness {features.mean_brightness:.1f} > {thresholds.cat_c_mean_high:.1f}"
        )
    if extreme_ratio > thresholds.cat_c_extreme_ratio:
        return "cat_c", (
            f"extreme_pixel_ratio {extreme_ratio:.3f} > {thresholds.cat_c_extreme_ratio:.3f}"
        )

    # --- cat_b: partial damage ---
    if features.laplacian_var < thresholds.cat_b_laplacian:
        return "cat_b", (
            f"laplacian_var {features.laplacian_var:.1f} < cat_b threshold "
            f"{thresholds.cat_b_laplacian:.1f}"
        )
    if extreme_ratio > thresholds.cat_b_extreme_ratio:
        return "cat_b", (
            f"extreme_pixel_ratio {extreme_ratio:.3f} > {thresholds.cat_b_extreme_ratio:.3f}"
        )

    # --- Temporal signal: upgrade cat_a → cat_b on anomalous brightness ---
    if (
        rolling_brightness_mean is not None
        and cfg.temporal_brightness_delta > 0.0
        and abs(features.mean_brightness - rolling_brightness_mean) > cfg.temporal_brightness_delta
    ):
        delta = features.mean_brightness - rolling_brightness_mean
        return "cat_b", (
            f"brightness_delta {delta:+.1f} (|{abs(delta):.1f}| > "
            f"{cfg.temporal_brightness_delta:.1f} from 5-frame rolling mean)"
        )

    return "cat_a", None


# ---------------------------------------------------------------------------
# Calibration report (written between passes for early inspection)
# ---------------------------------------------------------------------------

def compute_calibration(
    all_features: list[FrameFeatures],
    thresholds: ResolvedThresholds,
) -> dict:
    """Per-feature percentile distribution + resolved threshold values.

    Emit this before classification so threshold tuning can start while the
    run is in progress (or if it dies). Without this report, retuning is guesswork.
    """
    def _stats(arr: np.ndarray) -> dict:
        pcts = [0, 5, 25, 50, 75, 95, 100]
        vals = np.percentile(arr, pcts)
        keys = ["min", "p5", "p25", "median", "p75", "p95", "max"]
        return {k: round(float(v), 4) for k, v in zip(keys, vals)}

    lap = np.array([f.laplacian_var for f in all_features])
    mean_b = np.array([f.mean_brightness for f in all_features])
    dark = np.array([f.dark_pixel_ratio for f in all_features])
    bright = np.array([f.bright_pixel_ratio for f in all_features])
    return {
        "laplacian_var": _stats(lap),
        "mean_brightness": _stats(mean_b),
        "dark_pixel_ratio": _stats(dark),
        "bright_pixel_ratio": _stats(bright),
        "resolved_thresholds": {
            "cat_c_laplacian": round(thresholds.cat_c_laplacian, 4),
            "cat_b_laplacian": round(thresholds.cat_b_laplacian, 4),
            "cat_c_mean_low": thresholds.cat_c_mean_low,
            "cat_c_mean_high": thresholds.cat_c_mean_high,
            "cat_c_extreme_ratio": thresholds.cat_c_extreme_ratio,
            "cat_b_extreme_ratio": thresholds.cat_b_extreme_ratio,
        },
    }


# ---------------------------------------------------------------------------
# Gallery (static HTML for visual review)
# ---------------------------------------------------------------------------

def build_gallery_html(
    damage_entries: dict[str, dict],
    out_path: Path,
    max_per_cat: int = 10,
) -> None:
    """Generate a static HTML page with sampled cat-b and cat-c frames.

    Thumbnails reference frames_classified/ via relative path so the file
    is self-contained — open directly in any browser from the run directory.
    Seed is fixed (42) for deterministic sampling across re-runs.
    """
    rng = random.Random(42)
    cat_b = sorted(fn for fn, d in damage_entries.items() if d["category"] == "cat_b")
    cat_c = sorted(fn for fn, d in damage_entries.items() if d["category"] == "cat_c")
    sample_b = sorted(rng.sample(cat_b, min(max_per_cat, len(cat_b))))
    sample_c = sorted(rng.sample(cat_c, min(max_per_cat, len(cat_c))))

    def _card(fn: str) -> str:
        d = damage_entries[fn]
        cat = d["category"]
        s = d["scores"]
        reason = d.get("reason") or "\u2014"
        color = "#ff9900" if cat == "cat_b" else "#cc3300"
        return (
            f'<div style="border:3px solid {color};padding:8px;margin:6px;'
            f'display:inline-block;vertical-align:top;max-width:320px">'
            f'<img src="frames_classified/{fn}" style="max-width:300px;display:block">'
            f'<p style="font-family:monospace;font-size:11px;margin:4px 0">'
            f'<b>{fn}</b> [{cat}]<br>'
            f'lap={s["laplacian_var"]:.1f}&nbsp;&nbsp;bright={s["mean_brightness"]:.1f}<br>'
            f'dark={s["dark_pixel_ratio"]:.4f}&nbsp;&nbsp;extbright={s["bright_pixel_ratio"]:.4f}<br>'
            f'<span style="color:{color}">{reason}</span></p></div>'
        )

    sections: list[str] = []
    if sample_b:
        cards = "".join(_card(fn) for fn in sample_b)
        sections.append(
            f'<h2>Cat-B \u2014 Partial damage ({len(cat_b)} total, showing {len(sample_b)})'
            f'</h2>{cards}'
        )
    if sample_c:
        cards = "".join(_card(fn) for fn in sample_c)
        sections.append(
            f'<h2 style="color:#cc3300">Cat-C \u2014 Unusable ({len(cat_c)} total, '
            f'showing {len(sample_c)})</h2>{cards}'
        )

    body = "\n".join(sections) if sections else "<p>No damaged frames detected.</p>"
    html = (
        "<!doctype html><html><head><meta charset=utf-8>"
        "<title>S05 Damage Gallery</title>"
        "<style>body{font-family:sans-serif;background:#111;color:#eee;padding:16px}"
        "h2{margin-top:24px}</style></head><body>"
        "<h1>S05 Damage Gallery</h1>"
        f"{body}</body></html>"
    )
    out_path.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

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
            (c for c in candidates if any(d.name.startswith("s04_") for d in c.iterdir())),
            None,
        )
        if run_path is None:
            raise RuntimeError("No run with s04 output found. Run s04 first.")

    logger = get_logger("s05", run_path)
    logger.info("S05 damage_classify starting; run_dir=%s", run_path)

    in_frames_dir = stage_input_frames(run_path, "s05")
    frames = sorted(in_frames_dir.glob("*.png"))
    if not frames:
        raise RuntimeError(f"No frames in {in_frames_dir}")
    total = len(frames)
    logger.info("Input: %d frames from %s", total, in_frames_dir)

    # Load shot boundaries remapped to the movie stream (S04 output).
    # Fall back to single-shot if S04 was skipped (unusual).
    try:
        s04_dir = find_stage(run_path, "s04_")
        sba = json.loads((s04_dir / "shot_boundaries_adjusted.json").read_text())
        shot_boundaries: list[int] = sba.get("shot_boundaries", [])
    except FileNotFoundError:
        logger.warning("shot_boundaries_adjusted.json not found; treating film as single shot")
        shot_boundaries = []
    boundary_set = set(shot_boundaries)
    logger.info("Shot boundaries loaded: %d", len(shot_boundaries))

    s05cfg = cfg.s05_damage_classify
    out_dir = stage_dir(run_path, STAGE_ID, STAGE_NAME)
    frames_classified = out_dir / "frames_classified"
    if frames_classified.exists():
        shutil.rmtree(frames_classified)
    frames_classified.mkdir(parents=True)

    damage_map_frames: dict[str, dict] = {}

    with bench_run(f"{STAGE_ID}_{STAGE_NAME}", cfg.section_hash("s05"), out_dir) as b:

        if s05cfg.method == "passthrough":
            logger.info("Passthrough mode: all frames → cat_a, hardlinking.")
            for i, f in enumerate(frames):
                _link_or_copy(f, frames_classified / f.name)
                damage_map_frames[f.name] = {
                    "category": "cat_a",
                    "scores": {
                        "laplacian_var": 0.0,
                        "mean_brightness": 0.0,
                        "dark_pixel_ratio": 0.0,
                        "bright_pixel_ratio": 0.0,
                        "is_shot_boundary": i in boundary_set,
                    },
                    "reason": None,
                }

        else:
            # === Pass 1: parallel feature extraction ===
            # workers=0 means auto; use cpu_count on Mac, cap at 4 on Colab 2-vCPU env
            workers = s05cfg.workers or os.cpu_count() or 1
            logger.info(
                "Pass 1: feature extraction for %d frames (workers=%d)...", total, workers
            )
            t0 = time.perf_counter()
            with Pool(processes=workers) as pool:
                raw_results = pool.map(compute_features, frames)
            logger.info("Pass 1 done in %.1fs (%.1f fps)", time.perf_counter() - t0,
                        total / max(time.perf_counter() - t0, 1e-6))

            # Stable ordered list of names (Pool.map preserves input order)
            ordered_names = [f.name for f in frames]
            features_map: dict[str, FrameFeatures] = {name: feat for name, feat in raw_results}
            all_features = [features_map[name] for name in ordered_names]

            # === Between passes: resolve thresholds, emit calibration, compute rolling mean ===
            thresholds = resolve_thresholds(all_features, s05cfg)
            logger.info(
                "Resolved thresholds: cat_c_lap=%.2f  cat_b_lap=%.2f",
                thresholds.cat_c_laplacian, thresholds.cat_b_laplacian,
            )

            # Write calibration before classification so it's available even if run dies
            calib = compute_calibration(all_features, thresholds)
            (out_dir / "threshold_calibration.json").write_text(
                json.dumps(calib, indent=2), encoding="utf-8"
            )
            logger.info("threshold_calibration.json written")

            rolling_map = compute_rolling_brightness(
                ordered_names, features_map, shot_boundaries
            )

            # === Pass 2: classify (sequential; fast — O(n) dict lookups) ===
            logger.info("Pass 2: classifying %d frames...", total)
            for i, (name, feat) in enumerate(zip(ordered_names, all_features)):
                cat, reason = classify_frame(feat, thresholds, rolling_map.get(name), s05cfg)
                if reason is not None:
                    reason = f"{name}: {reason}"
                damage_map_frames[name] = {
                    "category": cat,
                    "scores": {
                        "laplacian_var": round(feat.laplacian_var, 4),
                        "mean_brightness": round(feat.mean_brightness, 4),
                        "dark_pixel_ratio": round(feat.dark_pixel_ratio, 6),
                        "bright_pixel_ratio": round(feat.bright_pixel_ratio, 6),
                        "is_shot_boundary": i in boundary_set,
                    },
                    "reason": reason,
                }
                if (i + 1) % 500 == 0:
                    logger.info("Classification: %d/%d", i + 1, total)

            # === Hardlink frames_classified (zero disk cost) ===
            logger.info("Hardlinking %d frames → frames_classified/", total)
            for f in frames:
                _link_or_copy(f, frames_classified / f.name)

        b.frames_processed = total

    # === Write damage_map.json (sorted keys for determinism across runs) ===
    damage_map = {
        "config_hash": cfg.section_hash("s05"),  # downstream stages use this for cache invalidation
        "total_frames": total,
        "frames": dict(sorted(damage_map_frames.items())),
    }
    (out_dir / "damage_map.json").write_text(json.dumps(damage_map, indent=2), encoding="utf-8")
    logger.info("damage_map.json written (%d entries)", total)

    # === Summary stats ===
    cats = [d["category"] for d in damage_map_frames.values()]
    n_a = cats.count("cat_a")
    n_b = cats.count("cat_b")
    n_c = cats.count("cat_c")
    cat_a_pct = n_a / total * 100
    cat_b_pct = n_b / total * 100
    cat_c_pct = n_c / total * 100
    logger.info(
        "Summary: cat_a=%d (%.1f%%)  cat_b=%d (%.1f%%)  cat_c=%d (%.1f%%)",
        n_a, cat_a_pct, n_b, cat_b_pct, n_c, cat_c_pct,
    )

    # === Gallery (heuristic mode only; passthrough has nothing interesting to show) ===
    if s05cfg.method != "passthrough":
        build_gallery_html(damage_map_frames, out_dir / "damage_gallery.html")
        logger.info("damage_gallery.html written")

    # === Auto-QC ===
    classified_count = len(list(frames_classified.glob("*.png")))
    if classified_count != total:
        raise RuntimeError(
            f"S05 auto-QC: frames_classified count {classified_count} != input {total}"
        )
    if len(damage_map_frames) != total:
        raise RuntimeError(
            f"S05 auto-QC: damage_map entries {len(damage_map_frames)} != input {total}"
        )

    # NaN check on score values
    for fname, d in damage_map_frames.items():
        s = d["scores"]
        for key in ("laplacian_var", "mean_brightness", "dark_pixel_ratio", "bright_pixel_ratio"):
            v = s[key]
            if v != v:  # NaN != NaN
                raise RuntimeError(f"S05 auto-QC: NaN in {key} for frame {fname}")

    # Hard stops (catastrophic thresholds)
    if cat_c_pct > 15.0:
        raise RuntimeError(
            f"S05 auto-QC (hard-stop): cat_c_pct {cat_c_pct:.1f}% > 15% — "
            "check thresholds or source damage. Refusing to continue."
        )
    if cat_a_pct < 50.0:
        raise RuntimeError(
            f"S05 auto-QC (hard-stop): cat_a_pct {cat_a_pct:.1f}% < 50% — "
            "most frames classified as damaged. Check classification logic."
        )

    # Soft fails (warnings, not stops)
    summary: dict = {
        "total_frames": total,
        "cat_a": n_a, "cat_a_pct": round(cat_a_pct, 2),
        "cat_b": n_b, "cat_b_pct": round(cat_b_pct, 2),
        "cat_c": n_c, "cat_c_pct": round(cat_c_pct, 2),
    }
    qc_soft_fail = False
    if cat_c_pct > 5.0:
        logger.warning(
            "S05 auto-QC (soft-fail): cat_c_pct %.1f%% > 5%% — "
            "review damage_gallery.html and tighten cat_c thresholds if needed", cat_c_pct,
        )
        qc_soft_fail = True
    if cat_a_pct < 70.0:
        logger.warning(
            "S05 auto-QC (soft-fail): cat_a_pct %.1f%% < 70%% — "
            "fewer healthy frames than expected", cat_a_pct,
        )
        qc_soft_fail = True
    if qc_soft_fail:
        summary["qc_soft_fail"] = True

    (out_dir / "damage_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    if qc_soft_fail:
        logger.warning("S05 complete with soft-fail(s). Review damage_gallery.html before proceeding.")
    else:
        logger.info("S05 auto-QC passed. Review gallery at damage_gallery.html.")


if __name__ == "__main__":
    main()
