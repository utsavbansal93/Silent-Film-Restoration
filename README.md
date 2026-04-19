# Silent Film Restoration

Modular pipeline for restoring surviving silent-era film fragments. This repo is framed as the first of a series; the initial project is **Lanka Dahan (1917)**, directed by Dadasaheb Phalke — ~5 min 16 s of surviving footage after the 2003 NFAI fire.

Source: [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Lanka_Dahan_(1917)_by_Dadasaheb_Phalke.webm) · 1920×1080 WebM · CC BY-SA 4.0.

Restoration register: closer to Peter Jackson's *They Shall Not Grow Old* than FIAF-purist preservation. Moderate-to-aggressive cleanup, 24 fps cap (no soap-opera interpolation), identity-preserving face restoration, faint grain retained for authenticity.

## Pass 1 scope (current)

- Repo scaffolding, config system, logging, JOURNAL
- S00 ingest (source → PNG sequence + metadata)
- S01 probe (shot boundaries, damage heuristics, intertitle detection via EAST)
- S02 stabilisation (Python `vidstab` OpenCV feature-tracking, per-shot, intertitles pass through)
- Local web viewer (source vs stabilised side-by-side scrubber)
- Test harness (unit + golden-frame + integration)
- Colab bootstrap notebook (with Kaggle fallback stub)

Later passes add: deflicker, damage classification, inpainting, denoise, retime, RIFE interpolation, Real-ESRGAN upscale, CodeFormer face restoration, sharpen, grain, grade, intertitle replacement, encode.

See [lanka_dahan_claude_code_brief.md](../lanka_dahan_claude_code_brief.md) for the full brief and [JOURNAL.md](JOURNAL.md) for every technical decision with rationale.

## Quickstart

```bash
# One-time setup
just bootstrap                # create venv, install deps
just fetch-source             # download Wikimedia WebM
just regenerate-fixture       # carve 30-second test clip

# Run the pipeline
just run modern_smooth        # full pipeline with primary profile
just run test_30sec           # 30-second fixture (fast iteration)

# Individual stages
just stage s00 modern_smooth
just stage s01 modern_smooth
just stage s02 modern_smooth

# Tests
just test                     # unit + golden + integration
just test-unit                # fast, no I/O

# Viewer
just viewer                   # serves on :8765 (auto-increments if taken)

# Benchmarks
just bench                    # re-run benchmarks, update BENCHMARKS.md
```

## Repo layout

```
Silent-Film-Restoration/
├── pipeline/
│   ├── common/           # config, logging, paths, hashing, bench, device
│   └── stages/           # s00_ingest.py, s01_probe.py, s02_stabilise.py
├── configs/              # schema.yaml, modern_smooth.yaml, test_30sec.yaml
├── viewer/               # server.py + index.html + app.js
├── tests/                # unit/ golden/ integration/ fixtures/
├── colab/                # bootstrap.ipynb (+ Kaggle stub)
├── scripts/              # regenerate_fixture.sh, fetch_source.sh
├── runs/                 # gitignored — timestamped per-run outputs
└── models/               # gitignored — downloaded model weights (EAST, etc.)
```

## Environments

Paths switch on `SILENT_FILM_ENV`:
- `local` (default) — `~/Documents/Claude Writing/Silent Restoration/Silent-Film-Restoration/`
- `colab` — `/content/drive/MyDrive/silent-film-restoration/lanka-dahan/`
- `kaggle` — `/kaggle/working/silent-film-restoration/lanka-dahan/`

## License

CC BY-SA 4.0. See [LICENSE](LICENSE). Matches Wikimedia source terms.
