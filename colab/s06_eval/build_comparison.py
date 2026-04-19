"""Render runs/s06-eval/comparison.html — 50 rows × 4 columns (input + 3 models).

Run locally after pulling Colab outputs from Drive into
runs/s06-eval/output/{bofbl,rrtn,deepremaster}/. Rows are grouped by
stratum (scratched → splotched → dirty → shot_edge → normal) to keep
similar damage types adjacent for easier comparison.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "runs" / "s06-eval"
MODELS = ["bofbl", "rrtn", "deepremaster"]
STRATUM_COLORS = {
    "scratched": "#ff5566",
    "splotched": "#ff9900",
    "dirty":     "#ffcc33",
    "shot_edge": "#33ccff",
    "normal":    "#66cc66",
}


def main() -> None:
    manifest = json.loads((OUT_DIR / "sample_frames.json").read_text())
    bench = {
        m: json.loads((OUT_DIR / "bench" / f"{m}.json").read_text())
        if (OUT_DIR / "bench" / f"{m}.json").exists() else None
        for m in MODELS
    }

    parts: list[str] = [
        "<!doctype html><html><head><meta charset=utf-8>",
        "<title>S06 eval — before/after (50 × 4)</title>",
        "<style>body{font-family:sans-serif;background:#111;color:#eee;padding:16px}",
        "h2{margin-top:24px}",
        "table{border-collapse:collapse;margin:8px 0}",
        "td{padding:4px;vertical-align:top;text-align:center}",
        "td img{max-width:260px;display:block}",
        ".lab{font-family:monospace;font-size:11px;padding:6px;color:#aaa}",
        ".bar{position:sticky;top:0;background:#222;padding:8px;border-bottom:1px solid #444}",
        "</style></head><body>",
        "<div class='bar'><h1 style='margin:0'>S06 eval — input vs BOFBL vs RRTN vs DeepRemaster</h1>",
        "<p class='lab'>",
    ]
    for m in MODELS:
        b = bench[m]
        if b is None:
            parts.append(f"{m}: (no bench)  &nbsp; ")
        else:
            parts.append(
                f"<b>{m}</b>: {b['n_frames']}/50 frames, {b['wall_s']}s, {b['fps']} fps &nbsp; "
            )
    parts.append("</p></div>")

    for stratum, items in manifest["details"].items():
        colour = STRATUM_COLORS.get(stratum, "#888")
        parts.append(
            f"<h2 style='color:{colour};border-left:6px solid {colour};padding-left:10px'>"
            f"{stratum} ({len(items)})</h2>"
        )
        parts.append("<table>")
        parts.append(
            "<tr>"
            "<td class='lab'>input</td>"
            "<td class='lab'>bofbl</td>"
            "<td class='lab'>rrtn</td>"
            "<td class='lab'>deepremaster</td>"
            "<td class='lab'>scores</td>"
            "</tr>"
        )
        for it in items:
            name = it["file"]
            row = [
                f"<img src='input/{name}'>",
                _img_or_missing(f"output/bofbl/{name}"),
                _img_or_missing(f"output/rrtn/{name}"),
                _img_or_missing(f"output/deepremaster/{name}"),
                (
                    f"<div class='lab'><b>{name}</b><br>"
                    f"lap={it['lap']:.1f}<br>brt={it['brightness']:.1f}<br>"
                    f"v={it['v_score']:.2f}<br>splotch={it['splotch']:.1f}<br>"
                    f"dust={it['dust']:.4f}</div>"
                ),
            ]
            parts.append("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>")
        parts.append("</table>")

    parts.append("</body></html>")
    out = OUT_DIR / "comparison.html"
    out.write_text("".join(parts))
    print(f"Wrote {out}")
    missing = {
        m: [it["file"] for items in manifest["details"].values() for it in items
            if not (OUT_DIR / "output" / m / it["file"]).exists()]
        for m in MODELS
    }
    for m, ms in missing.items():
        if ms:
            print(f"  {m}: missing {len(ms)}/50 outputs (first 3: {ms[:3]})")


def _img_or_missing(rel_path: str) -> str:
    abs_path = OUT_DIR / rel_path
    if abs_path.exists():
        return f"<img src='{rel_path}'>"
    return "<div class='lab' style='color:#666;padding:80px 20px'>— not run —</div>"


if __name__ == "__main__":
    main()
