# S06 dirt_remove — Phase A model evaluation

Three-way A/B of BOFBL, RRTN, DeepRemaster on 50 stratified cat_a frames,
before writing any pipeline code for S06.

## Why this exists (plain-language)

S06 removes dirt/scratches/stains from ~5,700 "intact-but-dirty" frames.
Three learned-restoration candidates exist; none were trained on 1917 B&W
Indian silent film. Phase A runs all three on 50 hand-picked frames so
Utsav can eyeball the before/after grid and pick a winner (or reject all
three and fall back to classical methods).

## Files

| File | Purpose |
|---|---|
| `sample_frames.py` | Picks 50 stratified cat_a frames (scratched / splotched / dirty / shot_edge / normal). Runs locally on M3. Writes `runs/s06-eval/sample_frames.json` + hardlinks inputs to `runs/s06-eval/input/`. |
| `build_sample_gallery.py` | Renders `runs/s06-eval/sample_gallery.html` for sanity-checking the 50 inputs before Colab. |
| `run_models.ipynb` | Colab notebook. One section per model. Restart runtime between sections for clean deps. Caches weights to Drive. Writes outputs to `runs/s06-eval/output/<model>/` on Drive. |
| `build_comparison.py` | Runs locally after Drive pull. Renders `runs/s06-eval/comparison.html` — 50 rows × 4 columns (input, BOFBL, RRTN, DeepRemaster). |

## Workflow

```bash
# 1. Pick samples (local; ~25 s on M3).
.venv/bin/python colab/s06_eval/sample_frames.py
.venv/bin/python colab/s06_eval/build_sample_gallery.py
open runs/s06-eval/sample_gallery.html         # eyeball

# 2. Push inputs to Drive (Drive connector or rclone).
#    Target: /MyDrive/silent-film-restoration/lanka-dahan/s06-eval/input/

# 3. Open colab/s06_eval/run_models.ipynb in Colab. Run each of the three
#    sections in order, restarting runtime between. ~30 min total on T4.

# 4. Pull outputs back from Drive into runs/s06-eval/output/.

# 5. Render comparison (local).
.venv/bin/python colab/s06_eval/build_comparison.py
open runs/s06-eval/comparison.html             # judgement call

# 6. Record decision in JOURNAL.md D20.
```

## Success criterion

Chosen model must show **≥60% of before/after pairs** as clearly improved,
without introducing new artifacts (hallucinated texture, smearing,
over-smoothing). Subjective, Utsav-judged. If no model clears the bar, we
reassess (classical scratch-detect + median-filter inpainting).

## Drive layout (Colab expects this)

```
/MyDrive/silent-film-restoration/lanka-dahan/
  s06-eval/
    input/                          # 50 PNGs
    weights/
      bofbl/                        # cached model checkpoints
      rrtn/
      deepremaster/
    output/
      bofbl/                        # 50 cleaned PNGs per model
      rrtn/
      deepremaster/
    bench/
      bofbl.json                    # wall time, peak mem, args
      rrtn.json
      deepremaster.json
```
