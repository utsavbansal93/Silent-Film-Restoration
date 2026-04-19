"""OCR orchestrator for Lanka Dahan (1917) intertitle cards.

Usage:
    python -m ocr.run --s04-dir runs/canonical-base/s04_intertitle_extract
"""
from __future__ import annotations

import base64
import json
import logging
import math
import sys
from datetime import datetime
from pathlib import Path

import click

from ocr.stages.s01_card_map import build_card_map
from ocr.stages.s02_ocr import make_reader, ocr_card

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
_EXPECTED_CARD_RANGE = (5, 30)


@click.command()
@click.option(
    "--s04-dir", "s04_dir", required=True, type=click.Path(exists=True, path_type=Path),
    help="Path to s04_intertitle_extract output directory.",
)
@click.option(
    "--out-dir", "out_dir", default=None, type=click.Path(path_type=Path),
    help="Per-run output directory. Defaults to ocr/runs/<timestamp>/.",
)
@click.option(
    "--canonical-dir", "canonical_dir", default=None, type=click.Path(path_type=Path),
    help="Stable canonical output directory. Defaults to ocr/canonical/.",
)
def main(s04_dir: Path, out_dir: Path | None, canonical_dir: Path | None) -> None:
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    if out_dir is None:
        out_dir = REPO_ROOT / "ocr" / "runs" / ts
    if canonical_dir is None:
        canonical_dir = REPO_ROOT / "ocr" / "canonical"

    out_dir.mkdir(parents=True, exist_ok=True)
    canonical_dir.mkdir(parents=True, exist_ok=True)

    logger.info("OCR run → %s", out_dir)

    # S01: build card map from S04 output
    logger.info("S01  building card map …")
    cards = build_card_map(s04_dir)
    logger.info("     %d cards: %s", len(cards), [c["card_id"] for c in cards])
    _write_json(out_dir / "card_map.json", _serialisable_card_map(cards))

    # S02: OCR each card
    logger.info("S02  initialising EasyOCR …")
    reader = make_reader()
    results: list[dict] = []
    for card in cards:
        logger.info("     OCR %s  (%s) …", card["card_id"], card["label"])
        result = ocr_card(card, reader)
        results.append(result)
        logger.info(
            "     → en=%.2f  hi=%.2f  notes=%r",
            result["english"]["confidence"],
            result["hindi"]["confidence"],
            result["notes"],
        )

    _write_json(out_dir / "intertitles.json", results)
    logger.info("Run JSON → %s", out_dir / "intertitles.json")

    # Auto-QC
    logger.info("QC   running checks …")
    _run_qc(results)

    # Report
    report_path = out_dir / "report.html"
    _write_report(results, cards, report_path)
    logger.info("Report → %s", report_path)

    # Canonical copy
    canonical_path = canonical_dir / "intertitles.json"
    _write_json(canonical_path, results)
    logger.info("Canonical → %s", canonical_path)

    # Also copy report to canonical
    (canonical_dir / "report.html").write_bytes(report_path.read_bytes())

    logger.info("Done.")


# ---------------------------------------------------------------------------
# Auto-QC
# ---------------------------------------------------------------------------

def _run_qc(results: list[dict]) -> None:
    errors: list[str] = []
    warnings: list[str] = []

    for r in results:
        cid = r["card_id"]

        if not r.get("source_frames"):
            errors.append(f"{cid}: source_frames is empty")

        has_text = r["marathi"]["text"] or r["hindi"]["text"] or r["english"]["text"]
        if not has_text:
            errors.append(f"{cid}: all language fields empty — no text extracted")

        for lang in ("marathi", "hindi", "english"):
            c = r[lang]["confidence"]
            if math.isnan(c) or math.isinf(c):
                errors.append(f"{cid}.{lang}: confidence is NaN/Inf")
            elif not (0.0 <= c <= 1.0):
                errors.append(f"{cid}.{lang}: confidence {c:.4f} outside [0, 1]")

    n = len(results)
    lo, hi = _EXPECTED_CARD_RANGE
    if not (lo <= n <= hi):
        warnings.append(
            f"Card count {n} is outside expected range [{lo}, {hi}] — "
            "likely reflects short surviving fragment (only ~5 min survives)"
        )

    for w in warnings:
        logger.warning("QC  WARN  %s", w)
    for e in errors:
        logger.error("QC  FAIL  %s", e)

    if errors:
        sys.exit(f"Auto-QC failed with {len(errors)} error(s). See log above.")

    logger.info("QC  PASS  (%d warning(s), 0 errors)", len(warnings))


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _write_report(results: list[dict], cards: list[dict], report_path: Path) -> None:
    card_lookup = {c["card_id"]: c for c in cards}

    rows: list[str] = []
    for r in results:
        cid = r["card_id"]
        card = card_lookup.get(cid, {})
        rep: Path | None = card.get("representative")

        img_tag = ""
        if rep and rep.exists():
            b64 = base64.b64encode(rep.read_bytes()).decode()
            img_tag = (
                f'<img src="data:image/png;base64,{b64}" '
                f'style="max-width:380px;max-height:280px;display:block;">'
            )

        def badge(conf: float) -> str:
            if conf >= 0.85:
                bg = "#16a34a"
            elif conf >= 0.55:
                bg = "#d97706"
            else:
                bg = "#dc2626"
            return (
                f'<span style="background:{bg};color:#fff;padding:1px 7px;'
                f'border-radius:4px;font-size:0.78em;font-family:monospace;">'
                f'{conf:.2f}</span>'
            )

        def lang_row(label: str, field: dict) -> str:
            text = field["text"] or "<span style='color:#9ca3af'>—</span>"
            conf = field["confidence"]
            highlight = "background:#fef3c7;" if conf < 0.55 and field["text"] else ""
            return (
                f'<tr style="{highlight}">'
                f'<td style="padding:4px 8px;font-weight:600;width:70px;">{label}</td>'
                f'<td style="padding:4px 6px;">{badge(conf)}</td>'
                f'<td style="padding:4px 8px;font-size:1.05em;">{text}</td>'
                f"</tr>"
            )

        notes_html = (
            f'<p style="color:#6b7280;font-size:0.82em;margin:6px 0 0;">'
            f'⚠ {r["notes"]}</p>'
            if r["notes"] else ""
        )

        rows.append(f"""
<div style="display:flex;gap:20px;border:1px solid #e5e7eb;border-radius:8px;
            padding:16px;margin-bottom:20px;align-items:flex-start;">
  <div style="flex:0 0 auto;">{img_tag}</div>
  <div style="flex:1;min-width:0;">
    <h3 style="margin:0 0 8px;font-size:1em;">
      {cid}
      <span style="color:#6b7280;font-weight:normal;font-size:0.88em;">
        {card.get("label", "")}
      </span>
    </h3>
    <table style="border-collapse:collapse;width:100%;">
      {lang_row("English", r["english"])}
      {lang_row("Hindi", r["hindi"])}
      {lang_row("Marathi", r["marathi"])}
    </table>
    <p style="color:#9ca3af;font-size:0.78em;margin:4px 0 0;">
      source: {", ".join(r["source_frames"])}
    </p>
    {notes_html}
  </div>
</div>""")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lanka Dahan (1917) — OCR Report</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    max-width: 1000px; margin: 40px auto; padding: 0 20px; color: #111;
    line-height: 1.5;
  }}
  h1 {{ border-bottom: 2px solid #111; padding-bottom: 8px; }}
  .legend {{ display:flex; gap:14px; margin-bottom:24px; font-size:0.85em; flex-wrap:wrap; }}
  .dot {{ width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:4px;vertical-align:middle; }}
  .note {{ background:#fef3c7;padding:8px 12px;border-radius:6px;font-size:0.85em;margin-bottom:20px; }}
</style>
</head>
<body>
<h1>Lanka Dahan (1917) — OCR Report</h1>
<div class="legend">
  <span><span class="dot" style="background:#16a34a;"></span>≥ 0.85  high</span>
  <span><span class="dot" style="background:#d97706;"></span>0.55 – 0.84  medium</span>
  <span><span class="dot" style="background:#dc2626;"></span>&lt; 0.55  low (row highlighted)</span>
</div>
<p class="note">
  <strong>Marathi field:</strong> EasyOCR cannot distinguish Hindi from Marathi at the script
  level — both use Devanagari. All Devanagari text is placed in the Hindi field.
  The Marathi field is intentionally empty; human review required to separate the two.
</p>
{"".join(rows)}
</body>
</html>"""

    report_path.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _serialisable_card_map(cards: list[dict]) -> list[dict]:
    out = []
    for c in cards:
        row = {}
        for k, v in c.items():
            if isinstance(v, Path):
                row[k] = str(v)
            elif isinstance(v, list) and v and isinstance(v[0], Path):
                row[k] = [str(f) for f in v]
            else:
                row[k] = v
        out.append(row)
    return out


if __name__ == "__main__":
    main()
