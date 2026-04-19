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
