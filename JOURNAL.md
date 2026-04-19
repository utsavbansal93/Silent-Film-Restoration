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
