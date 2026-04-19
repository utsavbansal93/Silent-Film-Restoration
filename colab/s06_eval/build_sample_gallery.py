"""Render runs/s06-eval/sample_gallery.html — mirrors damage_gallery.html layout.

Shows the 50 stratified input frames grouped by category, with per-frame
score annotations. Use this to sanity-check the sample BEFORE shipping it
to Colab for three-way BOFBL/RRTN/DeepRemaster inference.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "runs" / "s06-eval"

CATEGORY_COLORS = {
    "scratched": "#ff5566",
    "splotched": "#ff9900",
    "dirty":     "#ffcc33",
    "shot_edge": "#33ccff",
    "normal":    "#66cc66",
}


def main() -> None:
    with (OUT_DIR / "sample_frames.json").open() as f:
        manifest = json.load(f)

    parts: list[str] = [
        "<!doctype html><html><head><meta charset=utf-8>",
        "<title>S06 eval — stratified sample (50 frames)</title>",
        "<style>body{font-family:sans-serif;background:#111;color:#eee;padding:16px}",
        "h2{margin-top:24px}.tile{padding:8px;margin:6px;display:inline-block;vertical-align:top;max-width:320px}",
        "img{max-width:300px;display:block}",
        "p{font-family:monospace;font-size:11px;margin:4px 0}</style></head><body>",
        "<h1>S06 eval — stratified sample (50 frames)</h1>",
        f"<p>Generated from runs/canonical-base/s05_damage_classify/frames_classified/ via colab/s06_eval/sample_frames.py. Input for three-way BOFBL/RRTN/DeepRemaster evaluation.</p>",
    ]

    for cat, items in manifest["details"].items():
        colour = CATEGORY_COLORS.get(cat, "#888")
        parts.append(f'<h2 style="color:{colour}">{cat} ({len(items)})</h2>')
        for it in items:
            parts.append(
                f'<div class="tile" style="border:3px solid {colour}">'
                f'<img src="input/{it["file"]}">'
                f'<p><b>{it["file"]}</b><br>'
                f'lap={it["lap"]:.1f}&nbsp;&nbsp;brt={it["brightness"]:.1f}<br>'
                f'v={it["v_score"]:.2f}&nbsp;&nbsp;splotch={it["splotch"]:.1f}&nbsp;&nbsp;dust={it["dust"]:.4f}<br>'
                f'shot_boundary={it["is_shot_boundary"]}</p></div>'
            )

    parts.append("</body></html>")
    out = OUT_DIR / "sample_gallery.html"
    out.write_text("".join(parts))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
