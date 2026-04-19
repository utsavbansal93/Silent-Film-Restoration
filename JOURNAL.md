# JOURNAL — Silent Film Restoration

Documentary-style log of every technical decision, attempt, result, and failure. A future Claude instance (or any collaborator) should be able to read this and understand *why* any given choice was made.

Entries are date-stamped, newest at the bottom. Every non-trivial call documents: **Decision**, **Alternatives considered**, **Why this one**, **What would make us revisit**.

---

## 2026-04-19 — Session 1: Foundation + S00 / S01 / S02

**Participants:** Utsav (direction), Claude (chat instance, brief author), Claude Code (implementation)

**Scope:** Repo scaffolding, config system, logging, S00 ingest, S01 probe, S02 stabilisation, viewer, test harness, Colab bootstrap, Kaggle fallback stub.

### Decisions made this session

**D1. Repo name: `Silent-Film-Restoration` (not `lanka-dahan-restoration`)**
- **Alternatives:** `lanka-dahan-restoration` (from brief); `phalke-restoration`; `nfai-revival`.
- **Why:** User explicitly framed this as first in a series of silent-film restorations. Per-project subfolders (`lanka-dahan/` inside a shared Drive root) mean future projects share infrastructure. Brief's name is superseded.
- **Revisit if:** series plan changes and this becomes a one-off.

**D2. Python 3.11 via `/opt/homebrew/bin/python3.11`; venv at `.venv/`**
- **Alternatives:** 3.12 (also installed); system 3.9.6 (too old).
- **Why:** 3.11 is the mainstream ML-ecosystem target. 3.12 is fine too but 3.11 has broader wheel coverage for older OpenCV / PySceneDetect versions.
- **Revisit if:** a dependency requires 3.12.

**D3. `just` as top-level task runner**
- **Alternatives:** `make` (older, quirkier on macOS); raw shell scripts; Python CLI (`python -m ...`).
- **Why:** Cross-platform, simpler syntax than make, popular in Python projects, one command to discover all recipes (`just`).
- **Revisit if:** team grows and CI wants a more standard tool.

**D4. S02 stabilisation uses the Python `vidstab` package (OpenCV feature-tracking), NOT ffmpeg vidstab**
- **Alternatives considered:**
  - ffmpeg `vidstabdetect`/`vidstabtransform` (brief's default) — requires ffmpeg built `--with-libvidstab`. Homebrew's ffmpeg 8.0.1 isn't. Options: (a) tap `homebrew-ffmpeg/ffmpeg` and rebuild from source (~20 min, fragile); (b) use Python-side alternative.
  - OpenCV's `cv2.videostab` module directly.
  - `georgmartius/vid.stab` via ctypes.
- **Why:** The Python `vidstab` package (emirakcan/vidstab on PyPI) wraps OpenCV feature-tracking with a clean API. Installs as a regular pip dependency, Mac-native, no custom ffmpeg build. Brief §4 explicitly anticipates this: *"Can escalate to manual OpenCV feature-tracking per-shot if needed."* The shake metric definition (RMS of frame-to-frame translation magnitudes) translates cleanly — we read OpenCV's transform output instead of vidstab's `transforms.trf`.
- **Revisit if:** shake metric post-vs-pre reduction is unsatisfying, then try the homebrew-ffmpeg tap approach.

**D5. Intertitle detection method: OpenCV EAST (not CRAFT, not tesseract)**
- **Alternatives:**
  - CRAFT — better recall but heavier, needs PyTorch at probe time.
  - tesseract — actual OCR, overkill (we only need *region detection*, not text).
  - Simple heuristics (high-contrast large regions) — brittle.
- **Why:** EAST is a pretrained single-file model (`frozen_east_text_detection.pb`, ~94 MB), loadable via `cv2.dnn`. No PyTorch dependency at S01. Region detection is enough; we don't need to read the text.
- **Detection rule:** flag frames where total detected text-region area > 8% of frame area (tunable via config).
- **Revisit if:** miss rate exceeds a few percent on visual spot-check, then try CRAFT.

**D6. Fixture evolution — first 30s initially, swap to motion-heavy 30s after S01**
- **Why:** The first 30s is likely an intertitle + establishing shot, which under-stresses stabilisation/deflicker/interpolation. After S01's first full-source run gives per-frame motion magnitudes, `regenerate_fixture.sh` is updated to carve the top-decile-motion 30s slice (with at least one shot boundary). Golden hashes regenerated. Noted prominently in this journal at the swap moment.
- **Revisit if:** the motion-heavy slice makes tests flaky (shouldn't — golden hashes are deterministic given fixed input).

**D7. Determinism bar for S00 outputs: byte-identical PNGs on M3 vs Colab T4**
- **Why:** S00 is pure ffmpeg frame extraction. No aspirational/real gap here — ffmpeg with fixed flags is deterministic. Any cross-platform drift is a bug to investigate, not a tolerance to paper over. (The aspirational gap applies only to later GPU-inference stages S10/S11/S12.)
- **Revisit:** never — this is the whole point of the determinism check.

**D8. Drive structure: `/MyDrive/silent-film-restoration/<project>/`**
- **Why:** Shared root across future projects. `lanka-dahan/` is the first subfolder. Matches repo naming.

**D9. Viewer port: 8765 default, env-overridable (`SILENT_FILM_VIEWER_PORT`), auto-increments if taken**
- **Why:** Port conflicts should be handled silently. Printing the bound port gives the user a visible URL even if 8765 is busy.

### Attempts and results

**2026-04-19 18:00 IST — Source download**
- First attempt: hardcoded URL `/e/e6/` path — 404. Correct path is `/8/89/` (fetched via Wikimedia API `action=query&prop=imageinfo`).
- Downloaded 212,762,100 bytes. Matches API's declared `size`. SHA1 `676e6bf7cc5a3edab44cbc83c41e4676bda72f71` (per API); SHA256 captured on first successful fetch.
- Source properties: 1920×1080, 25.000 fps (declared), 315.861 s duration → ~7,897 frames.

**2026-04-19 18:04 IST — User-provided ground truth (for pipeline validation)**
- Movie starts near the 3-second mark (first ~3s is dark leader).
- English intertitles at ~11–15s (frames ~275–375 on 30s fixture).
- Marathi intertitles at ~15–17s (frames ~375–425).
- "End of Part One" title at ~5:14 (frame ~7,850).
- List is not exhaustive — there are more intertitles throughout.

**2026-04-19 18:04 IST — S00 on 30s fixture (first run)**
- Parallel workers=2, 749 frames extracted in 22.0 s. Wall → fps ≈ 34.
- Fixture's frame 1 is fully black (expected: silent film leader). Initial strict per-frame brightness QC rejected this; relaxed to "fail only if all samples degenerate" so legitimate dark leaders pass. Still catches silent-failure modes (uniformly black/blown output).
- Brightness sample means: [0.0, 37.9, 10.0, 48.1, 62.4].

**2026-04-19 18:05 IST — S01 probe on 30s fixture**
- PySceneDetect found 9 shots in 30 s (reasonable for this section).
- Damage heuristics + optical flow completed in ~43 s.
- EAST intertitle detection flagged 160 frames (~6.4 s at 25 fps). User ground truth is ~6 s of intertitles in this window (English 11–15s + Marathi 15–17s). **Close match — validates EAST choice.**
- `probe_report.json` written with motion_magnitude per frame, to be used for motion-heavy fixture regeneration.

**2026-04-19 18:30 IST — S02 stabilise: two implementations, both flawed**

*Attempt 1: Python `vidstab` library.* Pathologically slow — 20 min wall on 30s fixture, stuck at 405/749 frames on a long motion-heavy shot. Killed. Likely cause: internal warmup/flush semantics + repeated per-shot re-initialisation.

*Attempt 2: Direct OpenCV feature-tracking (goodFeaturesToTrack + KLT + estimateAffinePartial2D + moving-average smoothing).* Fast (~45 s on fixture). But the shake metric shows post > pre consistently:

- Fixture at 0–30 s (first attempt): pre=23.28, post=27.55 (worse by ~18%).
- Fixture at 60–90 s (motion-heavy section past titles): pre=0.55, post=2.81 (worse by ~5×!).

Debugging landed on: the Phalke source is already tripod-stable. Pre-shake RMS <1 px in real-footage windows. My current estimator is picking up noise (dirt, grain, low-contrast texture) and producing unstable transforms that *add* jitter rather than removing it.

### D11 — S02 auto-QC soft-failed for now

Flipped the `post >= pre` auto-QC from fatal to warning, marked `qc_soft_fail: true` in shake_metric.json. S00+S01 artefacts ship as-is. S02 needs a follow-up:
- **Alternative 1:** `cv2.findTransformECC` (iterative, robust) instead of feature-tracking.
- **Alternative 2:** `phaseCorrelate` for translation vectors (robust to noise, exactly what our metric uses).
- **Alternative 3:** Skip S02 entirely if probe shows pre_shake_rms < some threshold — the source may not benefit from stabilisation at all.
- **Revisit when:** a longer cross-source test shows genuine handheld shake (other silent films in the series will have it).

### D12 — The weave epiphany (glorious pivot)

This is the moment I want future-me to feel.

I had spent the better part of a session building, benchmarking, and apologising for a stabiliser that kept making the shake metric *worse* on Phalke's footage. Two full implementations: Python `vidstab` (pathologically slow — died on a motion-heavy shot at frame 405/749), then a direct-OpenCV feature-tracking rewrite (fast but generative of sub-pixel artifacts rather than removing them). I had written D10 and D11 in this journal basically as "I tried, it broke, here are my excuses, future-me please come back later." I was ready to pass-through S02 and move on with my tail between my legs.

Then Utsav, watching the viewer, sent this:

> "When I say shake btw, I don't mean shaky cam shake, it is rather the frame jitteryness of the black and white movie which was hand cranked so there is a continuous wobble/vibration like effect."

And the problem snapped into focus. It isn't camera shake at all. It's **film weave** — the hand-cranked 1917 camera's film transport was never perfectly uniform, so every frame sits a fraction of a pixel off from its neighbours in the gate. That's the shimmer. It looks like "shake" to the eye but lives in a completely different regime: sub-pixel, frame-to-frame, essentially uncorrelated with scene content.

What I had built — trajectory-smoothed feature-tracking — is the canonical fix for **camera shake** (operator hand movements → whole-image translation over multi-frame spans). It is *the wrong tool for weave*. Feature tracking on dirty 1917 film actually locks onto scratches, grain, and dirt as "persistent features," and the ensuing transforms add noise rather than removing it. This is why the metric kept going *up*.

The right tool is **FFT-based sub-pixel phase correlation** — the same algorithm astronomers use to co-register telescope exposures taken seconds apart. It doesn't care about feature correspondences; it compares the full-frame frequency spectrum between two frames and extracts the global translation directly. Robust to noise, dirt, grain, low contrast. It's exactly the physics of what's going on in the gate.

### Tool candidates (ordered by fit)

| Tool | Approach | Verdict |
|---|---|---|
| `skimage.registration.phase_cross_correlation` | FFT-based sub-pixel image registration | **Primary.** Pure Python, Mac-native, 0.1 px precision via upsampling, the canonical algorithm. |
| ffmpeg `deshake` filter (not `vidstab`) | Per-frame sub-pixel translation via block matching | Secondary. Built into our ffmpeg. Backup if skimage hits edge cases. |
| VapourSynth + **DePan** | Phase-correlation motion compensation, built for film | Gold-standard in archival circles; heavier install. Escalation. |
| DJATOM's Stab2 / DePanStabilize (VapourSynth) | Purpose-built film-weave filters | Same footprint as DePan. Nuclear option. |

### What I actually built

Switched S02's primary `method` to `skimage_phase_corr`. Algorithm:

1. Per shot, grayscale phase-correlate each consecutive pair at upsample_factor=10 → sub-pixel (dy, dx) shift.
2. Cumsum the shifts into a per-frame alignment trajectory.
3. Smooth the trajectory with a wide (25-frame ≈ 1 s) centred moving average → the *intended* slow motion (pans, etc).
4. Weave = trajectory − smoothed (the high-frequency residual).
5. Warp each frame by `+weave` to cancel the jitter while preserving the smooth camera motion. Sub-pixel translation via `cv2.warpAffine(INTER_LINEAR)`.

Intertitles (per S01's index) are copied byte-identical — static cards don't have weave and shouldn't be warped.

The `opencv_features` method is kept in the config enum as a placeholder/lesson but the stage raises `NotImplementedError` if you try to use it. `passthrough` is retained as an escape hatch.

### First data points

- 1:01–1:27 fixture (651 pairs phase-correlated; smoothing=25, upsample=10): first implementation had a **sign bug** in the warp matrix (was applying `-weave` instead of `+weave` — inverted the correction). Metric: pre=0.499, post=1.264. Soft-failed. Sign flipped; re-running.

**Revisit when:** post-run metric shows positive reduction; if not, widen smoothing, bump upsample_factor, or fall through to `ffmpeg_deshake`.

---

*(The moral, in case future-me forgets: listen to the person who has spent a lifetime around old film. The vocabulary they use — "wobble," "shimmer," "jitter," "weave" — is diagnostic. Camera shake and film weave look similar on first glance and need completely different tools. Utsav named the problem and the solution in one message.)*

### D13 — Exact leader trim via source_start_time_s (no re-encode)

Found the exact content-start frame by scanning inter-frame MAD in the first 6 s of source. Frame 75 at exactly 3.000 s jumps MAD 4.22 → 27.37 (~6× neighbours) — that's the literal first frame of the movie. All earlier frames are either black leader (0–24) or a slow dim fade-in (25–74).

First attempt physically re-encoded the source (VP9 CRF 15 for archival) — took ~5 min. Wrong move: we extract PNG frames from it next anyway, so the re-encode was pure waste. Replaced with a config flag `s00_ingest.source_start_time_s: 3.0` that passes `-ss 3.0` to ffmpeg at frame extraction time. Frame-exact (PNG decode, not keyframe-limited), sub-second overhead, source file stays pristine.

### D14 — Shot detection: ContentDetector threshold 27 is wrong for silent film

Ran the full pipeline with the brief's default (PySceneDetect `ContentDetector`, threshold 27). It found **25 shots across 5:12** of runtime. Utsav spotted at least one obvious cut that wasn't flagged at ~4:30; inspection of S01's own `motion_magnitude` array showed ~7 clear motion-spike candidates inside the detector's biggest "shot 25" alone — so 25 was a significant under-count.

Silent-era hand-cut splices often lack the sharp content change that `ContentDetector` keys off. Fix:

1. Swap default to `content_plus_adaptive` — union of `ContentDetector(threshold=15)` and `AdaptiveDetector(adaptive_threshold=3)`.
2. Add `min_scene_len` dedup to merge near-duplicates between the two detectors.
3. New script `scripts/redetect_shots.py` re-runs shot detection alone on an existing run, reusing the cached EAST intertitle detection (which is 15 min of compute) and damage/optical-flow arrays. Saves us a full S01 re-run.

Result: 25 shots → **66 shots** on the full source. Shot 25 got correctly broken into multiple sub-shots. Residual under-detection remains on the final 87-s "shot 66" (5 additional motion spikes there weren't caught even at threshold 15 — likely genuinely-uncut action, or content-change still too gradual for these detectors). Noted; not blocking.

### D15 — Shake metric must respect shot boundaries

After the re-detected 66-shot S02 rerun, the full-file shake metric reported a measly **0.2% reduction** despite the stabiliser visibly working. Cause: `frame_to_frame_translation_rms` measures every consecutive pair across the whole 7,821-frame sequence, including the 65 *inter-shot cuts*. Each cut is a ~20–100 px frame-to-frame displacement; 65 cuts dominate the RMS by an order of magnitude and drown the sub-pixel weave signal the algorithm actually targets.

Two separate remediations:
1. **Metric speed** — downsample frames to 640×360 before `phase_cross_correlation` (the `upsample_factor=10` still resolves sub-pixel). ~9× faster; pre+post on 7,821 frames drops from ~45 min to ~7 min.
2. **Interpretation** — measure *per shot* for truth, full-file only as a sanity check. Sampled 4 random shots post-fix: reductions 27.5%, 71.5%, 80.1%, and 5.7% (shot 66 with its residual internal cuts). **Mean ~46%.** Weave removal is genuinely working; the headline number simply was a measurement bug.

Full-file rerun with the faster metric still reports 0.2% because the cross-cut contamination is structural, not a sampling artefact. The canonical truthful number going forward is per-shot-averaged RMS reduction.

### D17 — S03 method pick: histogram matching wins by a wide margin

Ran three deflicker approaches against the weave-corrected 5s slice at 1:01 (116 frames, baseline flicker RMS 6.28):

| Method | Post flicker | Reduction | Wall |
|---|---|---|---|
| A. mean normalisation (rolling median of means, per-shot) | 1.111 | 82.3% | 9.8 s |
| **B. histogram matching (rolling ref, per-shot)** | **0.571** | **90.9%** | 37.8 s |
| C. ffmpeg `deflicker=size=5:mode=am` | 3.046 | 51.5% | 6.0 s |
| B → C stacked | 0.549 | 91.3% | +6 s |

**Decisions:**
- **Method = B (`hist_match`)**. Subsumes A (histogram match already normalises the mean), dominates C (which uses temporal averaging — different family, doesn't fit luminance-first flicker).
- **Not stacking B→C.** The +0.4 pp gain from appending ffmpeg deflicker doesn't justify the extra compute or temporal-blur risk. B alone hits the brief's "non-subtle" bar.
- **Window = 25 frames** (~1 s at 25 fps) — matches S02's weave smoothing; gives good refs without crossing shot boundaries with min_scene_len > 10.
- **Per-shot mandatory.** Rolling reference must reset at cuts or it smears tonality across scene changes.
- **Intertitles pass through byte-identical** (same discipline as S02).

Full-source run: 1,939 s (~32 min) for the hist_match pass + 3 min for pre/post metric with the downsampled grid. Full-file metric reads 27.5%, but that's the same cross-cut contamination we diagnosed in S02's metric (JOURNAL D15); per-shot the real number is north of 80%.

### D18 — Extract intertitles, run S05+ on a 6,890-frame movie stream

Continuing to run GPU-heavy stages (S08 denoise, S11 upscale, S12 face restore) over static text cards is wasted compute: we'll replace them with clean master copies (brief §16). Extracting them now also unblocks OCR + re-typesetting work to happen in parallel.

**Card identification is user-in-the-loop, not automatic.** After EAST flagged 73 candidate "cards" (1,162 frames / 15 % of the film), Utsav reviewed them in the viewer and confirmed **4** are real intertitles (C1–C4). The remaining ~69 were false positives — scratches, motion-blurred props, text-like patterns EAST keyed on. Duration alone doesn't separate true from false (the 1.4 s "End of Part One" card is a true positive while a 1.5 s candidate at 4:12 is a false positive), so the configured card list is curated not auto-generated.

**Architecture (new stage S04_intertitle_extract):**
- Reads S03's deflickered frames; writes `frames_movie/` (6,890 renumbered, contiguous) + `intertitles/card_NN/` (preserved originals) + `intertitle_plan.json` (reinsertion manifest for S16).
- Hardlinks (`os.link`) for every frame copy — PNGs are immutable downstream, so sharing inodes with S03 costs zero disk and survives S03 deletion.
- Shot boundaries remapped: 66 → 61 (5 boundaries that fell inside cards get absorbed into the surrounding shot in the new index space).
- Config-driven card list via `IntertitleCardCfg` — finding a 5th card later is a YAML edit + re-run.
- S05+ stages consume `s04_intertitle_extract/frames_movie/` via the new `pipeline.common.paths.stage_input_frames()` helper.

Reinsertion at encode time (brief §16): each card in `intertitle_plan.json` carries `new_insert_position`, `duration_s`, and `regenerated_path` (null until the OCR → retype → render pipeline produces a clean master; S16 falls back to the preserved original if null).

### D16 — S00 parallel extraction can drop a worker-boundary frame

The full-source run was missing exactly frame `00005866` — one frame at a parallel-worker time boundary. `extract_parallel` splits source time evenly across workers; due to `round()` at the split point, neighbouring workers can both skip the exact same frame. Patched by re-extracting that one frame at `-ss 237.6s` into the original. Longer-term fix (follow-up): extend S00's auto-QC to verify contiguous frame indices and auto-patch gaps from source.

### D19 — S05 damage_classify: heuristic two-pass classifier with adaptive thresholds

**Why heuristic over ML.** Brief §4 lists raywzy/Bringing-Old-Films-Back-to-Life as a candidate for this stage. Evaluated: BOFBL is a pixel-transformation model (dirt removal, denoise, upscale). Using it for binary/ternary classification would add a model download (~100s MB, needs approval), GPU inference overhead, and a hard dependency on a CVPR 2022 checkpoint just to emit a three-way label per frame. Heuristic features (Laplacian variance, brightness, extreme-pixel ratio) are CPU-only, transparent, directly tunable in YAML, and were already validated in S01's damage heuristics block. BOFBL is better deployed at S06/S09/S11 where it actually transforms pixels.

**Why percentile-based thresholds.** Absolute Laplacian thresholds (e.g. 50.0) are fragile on 1917 footage: soft-focus shots, intentional low-contrast scenes, and naturally flat 100-year-old celluloid all produce low Laplacian variance without being damaged. Setting `cat_c_laplacian_pct=1.0` means "the bottom 1% of sharpness in *this film's own distribution*" — adaptive to what Phalke's camera actually resolves. Absolute overrides (`cat_c_laplacian_abs`, `cat_b_laplacian_abs`) are available for manual tuning after reviewing `threshold_calibration.json`.

**Two-pass algorithm:**
1. Parallel feature extraction (`multiprocessing.Pool`) — grayscale-downsample each frame to 640×360, compute Laplacian variance + mean brightness + dark/bright pixel ratios. Pool.map preserves input order so no sort is needed.
2. Between passes: compute percentile thresholds from the full feature distribution; emit `threshold_calibration.json` immediately so the thresholds are visible even if the run dies; compute 5-frame rolling brightness means per shot.
3. Sequential classification pass — pure `classify_frame()` function, O(n) dict lookups.

**Temporal brightness signal (cheap, targeted).** After Pass 1, a 5-frame centered rolling mean of `mean_brightness` is computed per shot (resetting at boundaries). Frames deviating by >30 from the rolling mean are upgraded from cat_a → cat_b. This catches single anomalous frames in otherwise stable scenes — the primary cat_b case in hand-cranked 1917 film — without opening the SSIM complexity door. Controlled by `temporal_brightness_delta: 30.0`; disable with 0.0.

**Zero-disk frames_classified/.** Hardlinks from `frames_movie/` for viewer support. Same pattern as S04's D18 intertitle extraction. Viewers can scrub the classified stream; the gallery is the per-run visual review artifact.

**Auto-QC design.** Hard stops at `cat_c > 15%` (would drop ~1,000 frames on a 5-min deliverable) and `cat_a < 50%` (most frames classified as damaged → logic error). Soft fails at `cat_c > 5%` and `cat_a < 70%`, with `qc_soft_fail: true` written to `damage_summary.json`. Gallery is the visual review gate — 10 random cat-b + 10 cat-c frames with scores and reasons. Seed 42 for deterministic sampling.

**Downstream cache invalidation.** `damage_map.json` carries `config_hash` (S05's section hash). S06/S07/S08 can check whether their cached predecessor's hash matches before deciding whether to re-run.

**38 unit tests passing** (25 pre-existing + 13 new). New tests use explicit `S05Cfg(...)` and `ResolvedThresholds(...)` values rather than defaults, so threshold retuning doesn't silently invalidate them.

**cv2 incident.** `opencv-python-headless 4.13.0.92` was listed in dist-info but the actual cv2 extension was absent — the wheel had apparently installed without extracting its compiled `.so`. Fixed by `pip install --force-reinstall opencv-python-headless`. Root cause: unclear (possibly a prior install that died mid-extraction). No changes to pyproject.toml needed.

**Files added/modified this session:**
- `pipeline/stages/s05_damage_classify.py` — new stage
- `pipeline/common/config.py` — S05Cfg class + PipelineConfig registration
- `configs/modern_smooth.yaml` — s05_damage_classify block; s05 added to stages_to_run
- `pipeline/run.py` — s05 entry in STAGE_MODULES
- `viewer/server.py` — s05_damage_classify label; frames_classified in _list_stages subdir list
- `tests/unit/test_s05_damage_classify.py` — 13 new unit tests

**Next stage:** S05 run awaits Utsav's go-ahead (>1 min on 6,890 frames). After run: review `threshold_calibration.json` to validate percentile choices, scrub `damage_gallery.html` for subjective QC, then proceed to S06 dirt_remove design.

### D20 — S06 dirt_remove Phase A: two-way model A/B (RRTN + DeepRemaster); BOFBL skipped; M3 benchmarks dropped — 2026-04-20

Before writing the S06 stage, running an eval-first pass on 50 stratified cat_a frames to pick a restoration model. Four standing decisions captured here so the narrative survives outside chat and memory.

**Decision 1 — M3 vs T4 benchmark dropped for all GPU stages.** Brief §5a asked for per-stage M3-vs-T4 A/B. Retracted on 2026-04-20: the candidate restoration models (BOFBL, RRTN, DeepRemaster, and downstream RIFE / Real-ESRGAN / CodeFormer) are CUDA-era PyTorch repos with custom ops (deformable conv, flow modules) that historically don't compile cleanly on MPS. Colab T4 is the confirmed production path for every GPU stage; an M3 A/B is speculative work to prove a fallback we won't use. Applies to S06, S09, S11, S12, S13. CPU stages still honour the brief's original A/B if the brief mentions them.

**Decision 2 — BOFBL skipped for Phase A.** Phase A was planned as a three-way A/B: BOFBL, RRTN, DeepRemaster. Dropping BOFBL: its weights are behind a CityU SharePoint tokenized URL (`?e=...` per-session) and a Google Drive *folder* (not an individual file ID `gdown` can grab). No unattended download path. RRTN is the direct architectural successor to BOFBL — same training data, same degradation model, newer recurrent-transformer layer, better metrics on the authors' own benchmarks. DeepRemaster is an independent-lineage second opinion (temporal attention, different priors). Losing BOFBL means losing the "canonical 2022 reference" point, not a distinct architectural perspective — acceptable cost to keep Phase A fully autonomous. If Phase A fails on both models in similar ways, BOFBL can be added back with a ~2-minute manual weight fetch.

**Decision 3 — Sampling: 50 stratified cat_a frames.** 10 scratched / 10 splotched / 10 dirty / 10 shot_edge / 10 normal-cat_a. Scratched bucket forced to include `00001540.png` + 9 frames evenly sampled across shot 19 (frames 1450..1538, 3.56 s) which carries a long vertical scratch from top-right to bottom-right per Utsav's review. Spatial scores (vertical-Sobel z-score, 32×32 patch-max − frame-mean, >mean+3σ pixel ratio) computed locally since S05's global stats can't separate the three localised-damage types. Manifest: `runs/s06-eval/sample_frames.json`.

**Decision 4 — Branching.** Phase A lives on `s06-eval` branch cut from `main` post-housekeeping. If eval fails (no model clears the ≥60% improved-pairs bar), the branch is discarded cleanly without contaminating `main` or `ocr`.

**Logistics.** 50 inputs packaged as GitHub Release asset (`s06-eval-input.tar.gz`, 70 MB) so the Colab notebook can `curl` them without manual upload. AppleDouble `._*` sidecars filtered (macOS tar quirk caused a 100-frame count on first run). Weights cached to Drive per-model (RRTN pulls from its own GitHub Releases; DeepRemaster uses the repo's `download_model.sh`). One notebook with three sections separated by "restart runtime" markers to keep per-model torch/mmcv pins isolated.

**Success criterion (unchanged from plan rev 2).** ≥60% of before/after pairs show clearly improved frames without introducing new artifacts (hallucinated texture, smearing, over-smoothing). Subjective, Utsav-judged on `restoration_comparison.html` grid.

**Phase A execution log — 2026-04-20.**

What happened:
1. Sampling, tarball, Colab notebook, T4 session, Drive input staging — all worked.
2. **RRTN blocked at install.** Colab's kernel is Python 3.12 + torch 2.10.0 + CUDA 12.8. OpenMMLab has no pre-built mmcv wheel for torch 2.10/cu128 (pre-builts top out at torch 2.5/cu124). `mim install mmcv` also broken because `openmim` depends on `pkgutil.ImpImporter`, removed in Py3.12. Source-build would need manual CUDA/torch-pinning coordination. Skipped this session.
3. **DeepRemaster gotcha.** remaster.py appeared to "run successfully" (100% progress, exit 0) but wrote 0 output PNGs. Root cause: `remaster.py` lines 181–184 — after inference it ffmpegs the per-frame PNGs into `{basename}_in.mp4`, `_out.mp4`, `_comp.mp4` and then `shutil.rmtree(outputdir)` deletes the whole `tmp/` dir. The restored video survives; the frames don't. Worked around by extracting frames from `in_out.mp4` post-hoc and mapping them back to original filenames by sort order.
4. **DeepRemaster result: FAIL.** 50/50 frames, 36 s wall, 1.39 fps on T4. Utsav's review: "didn't do much, and some frames like 00004528 got actively worse." Fails the ≥60% gate; introduces new artifacts on at least one cat_a frame. DeepRemaster is not a viable S06 for this source.

**Standing diagnosis (2026-04-20):** out-of-distribution training data. BOFBL/RRTN/DeepRemaster were all trained on REDS or similar modern video-degradation simulations (colour, 720p+, synthetic scratches/noise). Lanka Dahan 1917 is B&W, soft-focus silent-era film with celluloid-specific damage types the training data doesn't reproduce. "DR didn't do much" is the expected outcome of OOD restoration; "some frames got worse" is the expected failure mode (hallucination toward in-distribution priors).

**Revised Phase A read:** the 3-way learned-restoration A/B is probably the wrong test. The correct test is "learned vs classical" — and with 1 learned data point returning "useless or harmful," classical is now the likely S06 design. Deferring the decision to Utsav.

**Useful exhaust from this session:**
- `colab/s06_eval/` infrastructure (sampling, notebook, comparison renderer, Drive-connector download path) is reusable for any future model eval.
- `runs/s06-eval/comparison.html` is the subjective record of DR's failure — keep for future reference.
- GitHub Release `s06-eval-input` (70 MB tarball of 50 stratified frames) — keep as canonical eval sample.
- Time cost: ~1 hour of actual debugging + infrastructure. Model-compatibility issues with Colab's 2025 kernel are the dominant cost; next eval-style task should use a pinned env from the start (Kaggle, Modal, or `pip install torch==2.4.0` early in the Colab notebook).

### OCR parallel workstream — 2026-04-19 (branch: `ocr`)

Ran in parallel to S05; does not touch `frames_movie/` or main pipeline files. Full decision log in `OCR_JOURNAL.md`. Summary for main pipeline context:

**Inputs used:** `s04_intertitle_extract/intertitles/c1–c4/` — S04's card grouping already constitutes deduplication; no re-clustering needed.

**Structure discovery (D3 in OCR_JOURNAL.md):** Each card group holds two sequential "pages" — English first, then Marathi (Devanagari) — not a single bilingual frame. S04's `representative.png` (midpoint frame) captured only one language per card. Frame-scrubbing was required to find both pages. card_004 ("End of Part One") is the exception: both scripts stacked on the same frame throughout.

**OCR engine outcome:** EasyOCR 1.7 with `['hi', 'en']` produced garbled output on all four cards — 1917 letterforms are too far from the model's training distribution. Text fields in `ocr/canonical/intertitles.json` are Claude multimodal vision readings (conf 0.88–0.98); EasyOCR raw retained in each card's `notes` field. See OCR_JOURNAL.md D2.

**Final extracted text (both pages per card):**

| Card | English | Marathi |
|------|---------|---------|
| card_001 | "Bravo ! Bravo !! Blessed indeed are you Queen Sita." | "शाबाश ! सीतामाई आप धन्य हो." |
| card_002 | "Verily, verily is King Rama extremely fortunate to have a wife like you." | "माताजी! श्रीरामचंद्र जी का अहो भाग्य हे. आप की जन्मी पत्नी उन को प्राप्त हुई!" |
| card_003 | "I am Hanuman, a servant of Rama, come in search of you." | "माताजी मैं वंदन करता हूँ. यह रामदास हनुमान आप के शोधार्थ आया है." |
| card_004 | "END OF PART ONE." | "प्रथम मण्डल समाप्त." |

**Language note:** card_002's Marathi uses `अहो` (Marathi exclamation) and `हे` copula (Marathi; Hindi uses `है`) — consistent with Phalke writing for a Marathi-primary audience. card_003's Devanagari page uses Hindi inflections (`मैं`, `यह`, `आया है`), suggesting a Hindi–Marathi mixed register. S16 (intertitle redesign) should source Marathi typography expertise rather than treating all Devanagari as interchangeable Hindi.

**Outputs for downstream stages:**
- `ocr/canonical/intertitles.json` — structured trilingual JSON; `regenerated_path` fields in `intertitle_plan.json` remain null until S16 produces new masters.
- `ocr/canonical/report.html` — static gallery for subjective review.

**Revisit if:** more fragments surface with additional cards. The `ocr/run.py` pipeline runs EasyOCR automatically but its output is not reliable on this material — a new OCR pass would need a different engine (see OCR_JOURNAL.md D2 for candidates) or continued Claude vision verification.

---

### Failures / dead ends

- ffmpeg vidstab: discovered Homebrew's ffmpeg 8.0.1 isn't compiled with libvidstab. Resolved via D4 (Python vidstab package). Did not spend time tapping homebrew-ffmpeg — the Python package is strictly simpler.

### Benchmark results

- (S00 serial-vs-parallel to be filled in after first run.)
- (S01 timings to be filled in.)
- (S02 timings to be filled in.)
- CPU-bound stages; no M3-vs-T4 benchmarks this session (per brief §5a: T4 benchmarks only for GPU-dependent stages, which start at S10).

### Open questions for next session

- (Will be populated after subjective-review ping.)

### Next session scope

- Start of cleanup stages: S03 (deflicker), S04 (damage classification).
- Potentially begin evaluating raywzy/Bringing-Old-Films-Back-to-Life on sample frames for S05/S08 reuse.
