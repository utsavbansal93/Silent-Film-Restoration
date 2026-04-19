# S06 dirt_remove — Phase A (learned-model eval on 50 stratified frames)

## Status

Phase A ran 2026-04-20. **No learned-restoration candidate succeeded on Phalke 1917 footage.** DeepRemaster was the only model to produce output (BOFBL skipped for weight-access reasons, RRTN blocked on mmcv wheel availability). DR's output was judged "didn't do much, some frames got worse" — fails the ≥60% improvement gate, introduces new artifacts. See JOURNAL D20.

Likely cause: out-of-distribution training data. BOFBL / RRTN / DeepRemaster all trained on modern colour video with synthetic degradations. 1917 B&W soft-focus celluloid is outside that distribution.

**Next eval candidate (not yet run):** Real-ESRGAN — different architecture family, pre-built PyPI wheels (no mmcv), has an `--outscale 1` mode for pure restoration without 4× upscale. Run via Colab terminal, same 50-frame inputs.

**Fallback plan:** if Real-ESRGAN also hurts frames, S06 goes classical (temporal-median scratch removal + top-hat dust detection + masked inpaint). Matches S05's heuristic approach per D19.

## Files

| File | Purpose |
|---|---|
| `sample_frames.py` | Picks 50 stratified cat_a frames. Runs locally on M3. Writes `runs/s06-eval/sample_frames.json` + hardlinks inputs to `runs/s06-eval/input/`. Scratched bucket is forced to include `00001540.png` + 9 frames from shot 19 (the 3.56 s scratch shot). |
| `build_sample_gallery.py` | Renders `runs/s06-eval/sample_gallery.html` — eyeball the 50 inputs before any model eval. |
| `build_comparison.py` | Renders `runs/s06-eval/comparison.html` — 4-column grid (input + bofbl + rrtn + deepremaster). Adds new model columns if their output dirs exist. |

Input samples are distributed as a GitHub release asset: `s06-eval-input.tar.gz` (70 MB, 50 PNGs) at https://github.com/utsavbansal93/Silent-Film-Restoration/releases/tag/s06-eval-input — Colab notebooks curl this rather than needing a manual upload.

## Drive layout

```
/MyDrive/silent-film-restoration/lanka-dahan/s06-eval/
  input/                  # 50 stratified PNGs (hardlinked from release tarball)
  weights/<model>/        # per-model checkpoint cache
  output/<model>/         # restored PNGs, mapped to original filenames
  bench/<model>.json      # wall time, fps, returncode, cmd
```

## Success gate (unchanged from plan rev 2)

Chosen model must show **≥60% of before/after pairs clearly improved without introducing new artifacts** (hallucinated texture, smearing, over-smoothing). Subjective, Utsav-judged.
