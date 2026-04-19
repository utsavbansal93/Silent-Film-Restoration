# Handoff note for the next Claude Code instance

Picking up on Utsav's Lanka Dahan (1917) restoration. This note is what I wish I'd had when I started. Read the brief at [../lanka_dahan_claude_code_brief.md](../lanka_dahan_claude_code_brief.md) first — it's the source of truth for the restoration philosophy. Read [JOURNAL.md](JOURNAL.md) for every technical decision made so far (D1 through D19). Memory at `/Users/utsavbansal/.claude/projects/-Users-utsavbansal-Documents-Claude-Writing-Silent-Restoration/memory/` has Utsav's working-style preferences (especially `feedback_confirm_before_heavy_jobs.md` — **read it**).

## Stage numbering — important change

The brief (§3) defined S00 → S17. We added an **extra stage** this session for intertitle extraction, which shifts everything from the brief's S04 onward by +1:

| Brief's number | Our number | Name |
|---|---|---|
| S00 | S00 | ingest |
| S01 | S01 | probe |
| S02 | S02 | stabilise (weave-removal, see D12) |
| S03 | S03 | deflicker |
| — | **S04** | **intertitle_extract** (new — see D18) |
| S04 | **S05** | damage_classify |
| S05 | **S06** | dirt_remove |
| S06 | **S07** | inpaint_partial |
| S07 | **S08** | drop_unusable |
| S08 | **S09** | denoise |
| S09 | **S10** | retime |
| S10 | **S11** | interpolate (RIFE) |
| S11 | **S12** | upscale (Real-ESRGAN) |
| S12 | **S13** | face_restore (CodeFormer) |
| S13 | **S14** | sharpen |
| S14 | **S15** | grain_add |
| S15 | **S16** | grade |
| S16 | **S17** | intertitle_replace |
| S17 | **S18** | encode |

When referring to the brief's stage list, mentally apply +1 from S05 onward. In code and config, always use our numbers.

## Where the work stands

**Completed and committed** (see https://github.com/utsavbansal93/Silent-Film-Restoration):

- **S00 ingest** — 7,821 PNGs from the trimmed source (`source_start_time_s: 3.0` in config — we skip a 3.0 s pure-black leader at `-ss` extraction time, no re-encode, see D13 + D15).
- **S01 probe** — 66 shots via union of `ContentDetector(threshold=15)` + `AdaptiveDetector` (D14). 1,162 EAST-flagged "intertitle" frames, of which only 4 cards are real (see below).
- **S02 stabilise** — sub-pixel film-weave removal via `skimage.registration.phase_cross_correlation`. Per-shot, wide smoothing (25-frame centred MA), upsample_factor=10. ~46% per-shot RMS reduction on sampled shots (D12 "glorious pivot", D15 metric caveat).
- **S03 deflicker** — per-shot rolling histogram matching. 91% flicker reduction on the 4.6s sample (D17). Full-file 27.5% reads the same cross-cut contamination we saw in S02's metric — real per-shot reduction is much higher.
- **S04 intertitle_extract** (new stage) — splits into a 6,890-frame `frames_movie/` stream + `intertitles/card_NN/` for OCR + `intertitle_plan.json` for reinsertion at S17. Hardlinks (`os.link`) for zero disk cost. 4 user-verified cards:
  - C1 frames 200–355 (6.24 s, at trimmed 0:08.00) — opening
  - C2 frames 3477–3834 (14.32 s, at trimmed 2:19.08) — mid-film
  - C3 frames 7055–7437 (15.32 s, at trimmed 4:42.20) — late-film
  - C4 frames 7774–7807 (1.36 s, at trimmed 5:10.96) — "End of Part One"
- **S05 damage_classify** — two-pass heuristic classifier (parallel feature extraction + percentile thresholds + temporal brightness signal). Run complete on canonical-base. Results:
  - cat_a: 5,707 (82.8%) — intact but dirty → S06
  - cat_b: 1,114 (16.2%) — partially corrupted → S07
  - cat_c: 69 (1.0%) — unusable → S08
  - Wall time: 30 s at 242 fps (8 workers). See D19.
- **Viewer** at `viewer/` with no-cache headers, autoplay, frame-position clamping to the intersection of "real" frames (for variant directories padded with symlinks/hardlinks).
- **38/38 unit tests green.**
- **Scripts** (not pipeline stages) — useful utilities:
  - `scripts/redetect_shots.py` — rerun S01 shot detection alone (reuses cached EAST)
  - `scripts/compute_shake_metric.py` — standalone S02 metric when S02 was interrupted mid-metric
  - `scripts/compare_s02_variants.py`, `scripts/compare_deflicker_methods.py`, `scripts/restructure_deflicker_variants.py` — one-off variant A/B harnesses kept for reference

**Next up:** S06 dirt_remove. Not started.

## Canonical run + disk state

- **Canonical run:** `runs/canonical-base/` — protected, don't delete.
- **Base reference frames** for final A/B comparison: `runs/canonical-base/s00_ingest/frames_raw/` (7,821 PNGs, ~6.5 GB). Never delete without user go-ahead.
- **S02 frames_stabilised/** has been deleted to free disk — recomputable from S00+S01 in ~11 min if needed. Its JSON outputs (bench, shake_metric) are still there. S03 and S04 don't depend on S02's PNGs (S03 wrote its own; S04 hardlinks to S03).
- **S05 frames_classified/** are hardlinks to S04's frames_movie/ — zero additional disk.
- **Disk:** ~25 GB free, ~88% used. S06 and later per-pixel stages each add ~10 GB of output PNGs — plan for this. We've already had one out-of-disk incident.

## Fragile state to know about

1. **Frame 5866 was originally dropped by S00 parallel extraction** (worker-boundary rounding bug, D16). Patched once via a one-off ffmpeg extract. If you re-run S00 fully, the bug will reappear unless you add a gap-check to `extract_parallel` — good follow-up.
2. **S04's `frames_movie/` and S05's `frames_classified/` are hardlinks, not copies.** Safe for downstream stages because they only *read* PNGs and write new files. Don't introduce any code path that overwrites files in these directories — it would corrupt S03 and the cards too (they share inodes). If you need to modify frames, write new files under a new stage dir.
3. **Intertitle metric vs per-shot metric:** the full-file flicker/shake RMS is polluted by inter-shot cut jumps. *Always* measure per-shot when reporting reductions to the user. See D15.
4. **The config's `s04_intertitle_extract.cards` is user-curated.** EAST has many false positives. Don't re-detect and auto-populate it — Utsav reviewed these in the viewer.
5. **Viewer labels in `viewer/server.py::_STAGE_LABELS`** — add an entry whenever you add a new stage output directory naming convention. Also extend the `frames_*` subdir list in `_list_stages` for the new stage.
6. **`damage_map.json` carries a `config_hash` field** — S06/S07/S08 should check this against the current S05 config hash before using cached results.
7. **00005972.png was manually reclassified** cat_c → cat_b directly in `damage_map.json` after Utsav's review. This is not reflected in the config or a re-run — it's a one-off patch. If S05 is re-run from scratch, this frame will revert to cat_c (lap=25.3 is barely below the p1 threshold of 29.9) and will need to be re-patched.

## S05 known limitations (relevant to S06 design)

The S05 classifier uses **global frame statistics** (Laplacian variance, mean brightness, pixel ratios). It **cannot detect localised damage** — vertical scratch lines, diagonal tears, white splotches. These are structurally intact frames where 1–5% of pixels are anomalous; the global averages don't move.

Utsav reviewed sample frames and confirmed these are present throughout:
- Thin vertical scratch lines (right-edge, common throughout)
- Diagonal white scratch lines
- White splotches (localised burns/stains)

**Current routing:** all such frames are cat_a → S06. This is fine IF:
- S06 (dirt_remove) can detect and fix them directly — which it should, since the candidate models (Bringing-Old-Films-Back-to-Life, RRTN) are trained on exactly these artefact types
- For frames where S06 can't fully clean them, the residual damage stays (S07 inpainting only runs on cat_b frames)

**S05 v2 enhancement (deferred):** add a spatial anomaly detector to catch these as cat_b:
- Divide downscaled frame into patches, find patches where local max >> local mean
- Threshold on anomalous patch count
- Only needed if S06 proves inadequate for splotches/large tears

Don't build this unless S06 testing reveals residual damage that S06 can't fix. The brief's philosophy is to not overbuild stages before knowing they're needed.

## Utsav's working style (critical)

Saved in memory; gist:

- He gives direction, not code. Readable + debuggable code is table-stakes.
- **"(just asking)" / "(just checking)" / "won't you …?" are questions, not approvals.** Never start long-running (>1 min, ≳1 GB, destructive) jobs on the strength of a parenthetical. Ask first. He got angry about this once — see `feedback_confirm_before_heavy_jobs.md` in memory.
- Tests proposed → approved → run → committed. In that order.
- JOURNAL.md is the story; keep writing there. D19 is the latest entry.
- Teach alongside: when you introduce a new term briefly explain it plain-English.
- Brief + polite + scientific. Disagree with him when you think he's wrong, but only once and with reasoning.

## How to resume

```bash
cd "/Users/utsavbansal/Documents/Claude Writing/Silent Restoration/Silent-Film-Restoration"

# Sanity
.venv/bin/pytest tests/unit/ -q          # expect: 38 passed

# Viewer
.venv/bin/python -m viewer.server        # http://127.0.0.1:8765/

# S05 outputs to review (already run)
cat runs/canonical-base/s05_damage_classify/damage_summary.json
cat runs/canonical-base/s05_damage_classify/threshold_calibration.json
# open runs/canonical-base/s05_damage_classify/damage_gallery.html in browser
```

## S06 dirt_remove — design considerations for next session

**What it does:** Remove dirt, scratches, and stains from cat_a frames. Cat_b frames are passed through (they go to S07 for inpainting). Cat_c frames are not processed (they go to S08 for dropping).

**Input:** `s05_damage_classify/frames_classified/` (same as `s04_intertitle_extract/frames_movie/`)
**Output:** `frames_cleaned/`
**Also reads:** `damage_map.json` for routing

**Primary candidate model: [raywzy/Bringing-Old-Films-Back-to-Life](https://github.com/raywzy/Bringing-Old-Films-Back-to-Life)** (CVPR 2022). Purpose-built for old film degradations. Handles structured noise (scratches, dirt) and unstructured noise simultaneously. Evaluate on sample frames first (the frames with visible scratch lines and splotches from the gallery).

**Secondary candidate: [mountln/RRTN-old-film-restoration](https://github.com/mountln/RRTN-old-film-restoration)** — builds on BOFBL with Recursive Recurrent Transformer Networks. Evaluate alongside.

**Key design questions to resolve:**
1. Does BOFBL/RRTN handle the specific artefacts in this source well? Eval on a 30-frame sample that includes known-scratched frames before committing.
2. GPU stage — needs M3-vs-T4 A/B benchmark per brief §5a.
3. Cat_b frames: pass through unchanged (S07 handles them) OR run BOFBL on them too before S07? Running BOFBL first reduces S07's inpainting workload.
4. Batch size auto-tune: start at 4, back off on OOM.

**Files to touch:**
- `pipeline/stages/s06_dirt_remove.py` — new stage (follow S05 shape)
- `pipeline/common/config.py` — add `S06Cfg`
- `configs/modern_smooth.yaml` — add `s06_dirt_remove:` block; append `s06` to `stages_to_run`
- `pipeline/run.py` — add `"s06"` to `STAGE_MODULES`
- `viewer/server.py` — add label; add `"frames_cleaned"` to subdir list
- `tests/unit/test_s06_dirt_remove.py` — unit tests (pure-function routing logic at minimum)
- `JOURNAL.md` — add D20

## Open questions to discuss with Utsav when he returns

- **S06 model evaluation first?** Before designing the full S06 stage, worth running BOFBL on 20–30 sample frames (including known-scratched ones) and showing Utsav the before/after. This is a ~10-minute Colab run with no pipeline plumbing needed. Gets subjective sign-off on the model before building the stage around it.
- **OCR pipeline for intertitle cards.** Orthogonal to restoration; can run in parallel. Brief §7.3 wants Devanagari typography sampled from Raja Harishchandra for Marathi/Hindi + period serif for English. Representative frames are at `runs/canonical-base/s04_intertitle_extract/intertitles/card_NN/representative.png`.
- **Full-source S02 metric** still reads 27.5% because of cross-cut contamination. Per-shot-averaged reduction report is a ~20-line extension to `scripts/compute_shake_metric.py` — worth doing once the pipeline stabilises.

## Files you'll touch most

- `pipeline/stages/s06_*.py` — the next stage you'll write. Follow the shape of `s05_damage_classify.py` (config-driven, writes `bench.json` + metric JSON, auto-QC). GPU stage — needs device detection and batch-size auto-tune.
- `pipeline/common/config.py` — add `S06Cfg`; register in `PipelineConfig`; add to `section_hash` map.
- `configs/modern_smooth.yaml` — add `s06_dirt_remove:` block; append `s06` to `stages_to_run`.
- `pipeline/run.py::STAGE_MODULES` — register.
- `viewer/server.py::_STAGE_LABELS` — add label; `frames_cleaned` to subdir list.
- `JOURNAL.md` — add D20 for S06 decision rationale.

Good luck. Keep the journal honest.
