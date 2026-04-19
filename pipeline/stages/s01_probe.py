"""S01 — Probe: characterise the source without mutating it.

Produces probe_report.json with:
- shot_boundaries: list of frame indices where shots change (PySceneDetect)
- damage: per-frame laplacian variance (sharpness) + dark-frame flag
- intertitles: list of frame indices flagged as intertitles (OpenCV EAST)
- motion_magnitude: per-frame RMS optical-flow magnitude (used later to pick
  the motion-heavy fixture slice)

EAST model is downloaded via scripts/fetch_east_model.sh if missing.
"""
from __future__ import annotations

import json
from pathlib import Path

import click
import cv2
import numpy as np

from pipeline.common.bench import bench_run
from pipeline.common.config import PipelineConfig, load_config
from pipeline.common.logging import get_logger
from pipeline.common.paths import new_run_dir, project_root, stage_dir

STAGE_ID = "s01"
STAGE_NAME = "probe"


def find_s00_dir(run_dir: Path) -> Path:
    for d in run_dir.iterdir():
        if d.is_dir() and d.name.startswith("s00_"):
            return d
    raise FileNotFoundError(f"No s00_* directory under {run_dir}")


def detect_shots(frames_dir: Path, threshold: float, logger) -> list[int]:
    """PySceneDetect on a frame sequence. Uses a virtual video from the sequence."""
    from scenedetect import SceneManager, open_video
    from scenedetect.detectors import ContentDetector

    frames = sorted(frames_dir.glob("*.png"))
    if not frames:
        return []
    # PySceneDetect can open an image sequence via ffmpeg backend by pointing to
    # the first frame with a pattern. Use the image-sequence path convention.
    digits = len(frames[0].stem)
    pattern = str(frames_dir / f"%0{digits}d.png")
    video = open_video(pattern)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video)
    scenes = scene_manager.get_scene_list()
    # Return frame indices at each boundary (start of each shot).
    boundaries = [s[0].get_frames() for s in scenes]
    logger.info("Detected %d shots (%d boundaries)", len(scenes), len(boundaries))
    return boundaries


def laplacian_variance(img_gray: np.ndarray) -> float:
    return float(cv2.Laplacian(img_gray, cv2.CV_64F).var())


def dark_ratio(img_gray: np.ndarray, threshold: int = 16) -> float:
    return float((img_gray < threshold).mean())


def compute_damage(frames: list[Path], logger) -> list[dict]:
    out = []
    for i, p in enumerate(frames):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            out.append({"frame": i, "error": "unreadable"})
            continue
        out.append({
            "frame": i,
            "lap_var": laplacian_variance(img),
            "dark_ratio": dark_ratio(img),
            "mean": float(img.mean()),
        })
        if (i + 1) % 500 == 0:
            logger.info("Damage heuristics: %d/%d frames", i + 1, len(frames))
    return out


def compute_motion(frames: list[Path], logger) -> list[float]:
    """Per-frame RMS optical-flow magnitude between frame i and i-1."""
    prev = None
    motions: list[float] = [0.0]  # first frame has no predecessor
    for i, p in enumerate(frames):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            motions.append(0.0)
            continue
        # Downsample for speed.
        small = cv2.resize(img, (320, 180))
        if prev is None:
            prev = small
            continue
        flow = cv2.calcOpticalFlowFarneback(
            prev, small, None, 0.5, 3, 15, 3, 5, 1.2, 0,
        )
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        motions.append(float(np.sqrt((mag ** 2).mean())))
        prev = small
        if (i + 1) % 500 == 0:
            logger.info("Optical flow: %d/%d frames", i + 1, len(frames))
    return motions


def load_east(model_path: Path, logger):
    if not model_path.exists():
        raise FileNotFoundError(
            f"EAST model missing at {model_path}. Run: just fetch-east-model"
        )
    logger.info("Loading EAST model: %s", model_path)
    net = cv2.dnn.readNet(str(model_path))
    return net


def east_text_area_fraction(
    net, img_bgr: np.ndarray, resize_w: int, resize_h: int, min_confidence: float,
) -> float:
    """Return fraction of the frame covered by detected text regions."""
    H, W = img_bgr.shape[:2]
    resized = cv2.resize(img_bgr, (resize_w, resize_h))
    blob = cv2.dnn.blobFromImage(
        resized, 1.0, (resize_w, resize_h), (123.68, 116.78, 103.94), swapRB=True, crop=False,
    )
    net.setInput(blob)
    scores, geometry = net.forward([
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3",
    ])
    num_rows, num_cols = scores.shape[2:4]
    total_area = 0.0
    for y in range(num_rows):
        scores_row = scores[0, 0, y]
        x0 = geometry[0, 0, y]
        x1 = geometry[0, 1, y]
        x2 = geometry[0, 2, y]
        x3 = geometry[0, 3, y]
        angles = geometry[0, 4, y]
        for x in range(num_cols):
            if scores_row[x] < min_confidence:
                continue
            h = x0[x] + x2[x]
            w = x1[x] + x3[x]
            total_area += float(h * w)
    # Scale back to original frame. EAST boxes are in the resized space.
    scale = (W / resize_w) * (H / resize_h)
    return (total_area * scale) / (W * H)


def detect_intertitles(frames: list[Path], cfg, logger) -> list[int]:
    model_path = project_root() / cfg.model_path
    net = load_east(model_path, logger)
    flagged: list[int] = []
    for i, p in enumerate(frames):
        img = cv2.imread(str(p))
        if img is None:
            continue
        frac = east_text_area_fraction(
            net, img, cfg.resize_width, cfg.resize_height, cfg.min_confidence,
        )
        if frac >= cfg.area_threshold:
            flagged.append(i)
        if (i + 1) % 200 == 0:
            logger.info("EAST intertitle detection: %d/%d (flagged so far: %d)",
                        i + 1, len(frames), len(flagged))
    return flagged


@click.command()
@click.option("--config", "cfg_path", required=True)
@click.option("--run-dir", "run_dir", default=None)
def main(cfg_path: str, run_dir: str | None) -> None:
    cfg = load_config(cfg_path)

    if run_dir:
        run_path = Path(run_dir)
    else:
        # Find most recent run with an s00 output.
        from pipeline.common.paths import runs_dir
        candidates = sorted(runs_dir().iterdir(), reverse=True)
        run_path = next((c for c in candidates if any(d.name.startswith("s00_") for d in c.iterdir())), None)
        if run_path is None:
            raise RuntimeError("No run with s00 output found. Run s00 first.")

    logger = get_logger("s01", run_path)
    logger.info("S01 probe starting; run_dir=%s", run_path)

    s00 = find_s00_dir(run_path)
    frames_dir = s00 / "frames_raw"
    frames = sorted(frames_dir.glob("*.png"))
    if not frames:
        raise RuntimeError(f"No frames in {frames_dir}")
    logger.info("Found %d frames", len(frames))

    out_dir = stage_dir(run_path, STAGE_ID, STAGE_NAME)

    with bench_run(f"{STAGE_ID}_{STAGE_NAME}", cfg.section_hash("s01"), out_dir) as b:
        b.frames_processed = len(frames)

        logger.info("Running shot detection...")
        shots = detect_shots(frames_dir, cfg.s01_probe.shot_detection.threshold, logger)

        logger.info("Running damage heuristics...")
        damage = compute_damage(frames, logger)

        logger.info("Running optical flow for motion magnitudes...")
        motion = compute_motion(frames, logger)

        intertitles: list[int] = []
        if cfg.s01_probe.intertitle_detection.enabled:
            logger.info("Running EAST intertitle detection...")
            try:
                intertitles = detect_intertitles(frames, cfg.s01_probe.intertitle_detection, logger)
            except FileNotFoundError as e:
                logger.error(str(e))
                logger.error("Skipping intertitle detection for this run.")

    report = {
        "frame_count": len(frames),
        "shot_boundaries": shots,
        "shot_count": len(shots),
        "damage": damage,
        "motion_magnitude": motion,
        "intertitle_frames": intertitles,
        "intertitle_count": len(intertitles),
    }
    (out_dir / "probe_report.json").write_text(json.dumps(report, indent=2))
    logger.info("Wrote probe_report.json: %d shots, %d intertitle frames",
                len(shots), len(intertitles))

    # Auto-QC
    if not (1 <= len(shots) <= 100):
        raise RuntimeError(f"S01 auto-QC: implausible shot count {len(shots)}")
    if len(motion) != len(frames):
        raise RuntimeError("S01 auto-QC: motion array length != frame count")
    if any(d.get("error") for d in damage):
        logger.warning("S01: some frames were unreadable")
    logger.info("S01 auto-QC passed")


if __name__ == "__main__":
    main()
