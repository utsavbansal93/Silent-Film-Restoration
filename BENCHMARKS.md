# Benchmarks

One table per stage. Rolling — append rows, don't delete history. Columns:

- `date` — ISO date of the run
- `device` — `mps/apple_m3`, `cuda/tesla_t4`, `cpu` (+ worker count for parallel runs)
- `input` — `fixture_30s` or `full_source_5m16s`
- `fps` — frames processed per second (end-to-end wall clock, not model-only)
- `wall_time_s` — seconds
- `peak_mem_mb` — RSS for CPU stages, GPU mem for GPU stages
- `notes` — config hash prefix, batch size, anomalies

Empty tables are seeded; first row lands after the stage runs on the 30s fixture or full source.

---

## S00 ingest

| date | device | input | fps | wall_time_s | peak_mem_mb | notes |
|------|--------|-------|-----|-------------|-------------|-------|

## S01 probe

| date | device | input | fps | wall_time_s | peak_mem_mb | notes |
|------|--------|-------|-----|-------------|-------------|-------|

## S02 stabilise

| date | device | input | fps | wall_time_s | peak_mem_mb | notes |
|------|--------|-------|-----|-------------|-------------|-------|

---

## Cross-stage: M3 vs T4 recommendation

| stage | recommended platform | margin | decided on |
|-------|----------------------|--------|------------|
| s00 | M3 (I/O-bound, no GPU help) | — | 2026-04-19 |
| s01 | M3 (CPU, EAST runs fine on CPU) | — | 2026-04-19 |
| s02 | M3 (CPU, OpenCV feature-tracking) | — | 2026-04-19 |
