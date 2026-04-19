# Handoff note for the next Claude Code instance

Picking up on Utsav's Lanka Dahan (1917) restoration. This note is what I wish I'd had when I started. Read the brief at [../lanka_dahan_claude_code_brief.md](../lanka_dahan_claude_code_brief.md) first — it's the source of truth for the restoration philosophy. Read [JOURNAL.md](JOURNAL.md) for every technical decision made so far (D1 through D18). Memory at `/Users/utsavbansal/.claude/projects/-Users-utsavbansal-Documents-Claude-Writing-Silent-Restoration/memory/` has Utsav's working-style preferences (especially `feedback_confirm_before_heavy_jobs.md` — **read it**).

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

Why the shift: intertitle cards are going to be replaced with clean master copies at encode time (brief §7.2). Running them through the GPU-heavy stages (denoise, upscale, face-restore) is wasted compute. Extracting them before the pixel-heavy pipeline starts was the right architectural move, and adding an extra stage was the cleanest way to represent it without renaming pre-existing ones.

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
- **Viewer** at `viewer/` with no-cache headers, autoplay, frame-position clamping to the intersection of "real" frames (for variant directories padded with symlinks/hardlinks).
- **25/25 unit tests green.**
- **Scripts** (not pipeline stages) — useful utilities:
  - `scripts/redetect_shots.py` — rerun S01 shot detection alone (reuses cached EAST)
  - `scripts/compute_shake_metric.py` — standalone S02 metric when S02 was interrupted mid-metric
  - `scripts/compare_s02_variants.py`, `scripts/compare_deflicker_methods.py`, `scripts/restructure_deflicker_variants.py` — one-off variant A/B harnesses kept for reference

**Next up:** S05 damage_classify. Not started. Utsav paused the session here; last message asked me to write this handoff note.

## Canonical run + disk state

- **Canonical run:** `runs/canonical-base/` — protected, don't delete.
- **Base reference frames** for final A/B comparison: `runs/canonical-base/s00_ingest/frames_raw/` (7,821 PNGs, ~6.5 GB). Never delete without user go-ahead.
- **S02 frames_stabilised/** has been deleted to free disk — recomputable from S00+S01 in ~11 min if needed. Its JSON outputs (bench, shake_metric) are still there. S03 and S04 don't depend on S02's PNGs (S03 wrote its own; S04 hardlinks to S03).
- **Disk:** 25 GB free, 88 % used at handoff. S05 and later per-pixel stages will each eat ~10 GB of output PNGs — plan for this. We've already had one out-of-disk incident.

## Fragile state to know about

1. **Frame 5866 was originally dropped by S00 parallel extraction** (worker-boundary rounding bug, D16). Patched once via a one-off ffmpeg extract. If you re-run S00 fully, the bug will reappear unless you add a gap-check to `extract_parallel` — good follow-up.
2. **S04's `frames_movie/` is hardlinks, not copies.** Safe for downstream stages because they only *read* PNGs and write new files. Don't introduce any code path that overwrites files in `frames_movie/` — it would corrupt S03 and the cards too (they share inodes). If you need to modify frames, write new files under a new stage dir.
3. **Intertitle metric vs per-shot metric:** the full-file flicker/shake RMS is polluted by inter-shot cut jumps. *Always* measure per-shot when reporting reductions to the user. See D15.
4. **The config's `s04_intertitle_extract.cards` is user-curated.** EAST has many false positives. Don't re-detect and auto-populate it — Utsav reviewed these in the viewer.
5. **Viewer labels in `viewer/server.py::_STAGE_LABELS`** — add an entry whenever you add a new stage output directory naming convention. Also extend the `frames_*` subdir list in `_list_stages` for the new stage.

## Utsav's working style (critical)

Saved in memory; gist:

- He gives direction, not code. Readable + debuggable code is table-stakes.
- **"(just asking)" / "(just checking)" / "won't you …?" are questions, not approvals.** Never start long-running (>1 min, ≳10 GB, destructive) jobs on the strength of a parenthetical. Ask first. He got angry about this once in this session — see `feedback_confirm_before_heavy_jobs.md` in memory.
- Tests proposed → approved → run → committed. In that order.
- JOURNAL.md is the story; keep writing there. D18 is the latest entry.
- Teach alongside: when you introduce a new term (MAD, phase correlation, hardlinks, etc.) briefly explain it plain-English.
- Brief + polite + scientific. Disagree with him when you think he's wrong, but only once and with reasoning.

## How to resume

```bash
cd "/Users/utsavbansal/Documents/Claude Writing/Silent Restoration/Silent-Film-Restoration"

# Sanity
.venv/bin/pytest tests/unit/ -q          # expect: 25 passed

# Viewer (also restarts after any server.py edit — no-cache headers mean
# plain refresh works, no Cmd+Shift+R needed)
.venv/bin/python -m viewer.server        # http://127.0.0.1:8765/

# If you want to see state of the canonical run
ls runs/canonical-base/
cat runs/canonical-base/s04_intertitle_extract/intertitle_plan.json
```

## Open questions to discuss with Utsav when he returns

- **S05 damage_classify design** — not proposed yet. Brief §3.4b: three categories (intact-but-dirty, torn/stained, fully-unusable). Our S05 will operate on `s04_intertitle_extract/frames_movie/` (the 6,890-frame intertitle-free stream), not the original S03 output. Shot boundaries come from `shot_boundaries_adjusted.json`, not `probe_report.json`.
- **OCR pipeline for intertitle cards.** Orthogonal to restoration; can run in parallel. Brief §7.3 wants Devanagari typography sampled from Raja Harishchandra for Marathi/Hindi + period serif for English. Representative frames are at `runs/canonical-base/s04_intertitle_extract/intertitles/card_NN/representative.png`.
- **Full-source S02 metric still reads 27.5 % because of cross-cut contamination.** We have the per-shot script (`scripts/compute_shake_metric.py`) but it computes file-level. A proper per-shot-averaged reduction report would be a ~20-line script extending it — worth doing once the pipeline stabilises.

## Files you'll touch most

- `pipeline/stages/s05_*.py` — the next stage you'll write. Follow the shape of `s03_deflicker.py` (per-shot, config-driven, writes `bench.json` + a stage-specific metric JSON, auto-QC).
- `pipeline/common/config.py` — add `S05Cfg`; register in `PipelineConfig`; add to `section_hash` map.
- `pipeline/common/paths.py::stage_input_frames` — already understands the S05+ → `frames_movie/` edge.
- `configs/modern_smooth.yaml` — add `s05_damage_classify:` block; append `s05` to `stages_to_run`.
- `pipeline/run.py::STAGE_MODULES` — register.
- `viewer/server.py::_STAGE_LABELS` — add label.
- `JOURNAL.md` — add D19 for S05 decision rationale.

Good luck. Keep the journal honest.
