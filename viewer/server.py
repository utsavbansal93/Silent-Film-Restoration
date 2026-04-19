"""Local viewer HTTP server. Source-vs-stabilised side-by-side scrubber.

Default port 8765; override with SILENT_FILM_VIEWER_PORT; auto-increments if taken.
Serves viewer/ assets plus runs/ frame directories as a file tree.
"""
from __future__ import annotations

import http.server
import json
import os
import socket
import socketserver
from pathlib import Path

from pipeline.common.paths import project_root, runs_dir


def find_port(start: int) -> int:
    for p in range(start, start + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    raise RuntimeError(f"No free port in {start}-{start + 20}")


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(project_root()), **kwargs)

    def do_GET(self):  # noqa: N802
        if self.path == "/api/runs":
            self._send_json(self._list_runs())
            return
        if self.path.startswith("/api/stages/"):
            run_name = self.path.split("/api/stages/", 1)[1]
            self._send_json(self._list_stages(run_name))
            return
        if self.path.startswith("/api/frames/"):
            rest = self.path.split("/api/frames/", 1)[1]
            try:
                run_name, stage_dir = rest.split("/", 1)
            except ValueError:
                self.send_error(400); return
            self._send_json(self._list_frames(run_name, stage_dir))
            return
        if self.path == "/" or self.path == "/index.html":
            self.path = "/viewer/index.html"
        return super().do_GET()

    def _send_json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _list_runs(self):
        runs = []
        for d in sorted(runs_dir().iterdir(), reverse=True):
            if not d.is_dir():
                continue
            # Friendly label: "1 Apr 19, 19:03 — test_30sec"
            # from run dirname "2026-04-19_190311_test_30sec"
            parts = d.name.split("_", 2)
            if len(parts) == 3:
                date, hms, profile = parts
                pretty_time = f"{hms[:2]}:{hms[2:4]}:{hms[4:6]}" if len(hms) == 6 else hms
                label = f"{date} {pretty_time} — {profile}"
            else:
                label = d.name
            runs.append({"name": d.name, "label": label})
        return runs

    _STAGE_LABELS = {
        "s00_ingest": "S00 — Raw (ingested)",
        "s02_stabilise": "S02 — Stabilised (weave-removed)",
        "s02a_default": "S02a — Default (smooth=25, upsample=10)",
        "s02b_wide": "S02b — Wider smoothing (smooth=75, upsample=10)",
        "s02c_fine": "S02c — Finer sub-pixel (smooth=25, upsample=50)",
        "s02d_ffmpeg": "S02d — ffmpeg deshake (block matching)",
        "s02e_median": "S02e — Default + temporal-median second pass (stacked)",
        "s03_deflicker": "S03 — Deflickered",
        "s05_dirt_remove": "S05 — Dirt removed",
        "s08_denoise": "S08 — Denoised",
        "s10_interpolate": "S10 — Interpolated (24 fps)",
        "s11_upscale": "S11 — Upscaled",
        "s12_face_restore": "S12 — Faces restored",
        "s14_grain_add": "S14 — Grain re-added",
        "s15_grade": "S15 — Graded",
    }

    def _list_stages(self, run_name: str):
        run = runs_dir() / run_name
        if not run.is_dir():
            return []
        stages = []
        for d in sorted(run.iterdir()):
            if not d.is_dir():
                continue
            for sub in ("frames_raw", "frames_stabilised", "frames_deflickered",
                        "frames_cleaned", "frames_inpainted", "frames_kept",
                        "frames_denoised", "frames_retimed", "frames_24fps",
                        "frames_upscaled", "frames_faces", "frames_sharpened",
                        "frames_final", "frames_graded", "frames_with_titles"):
                if (d / sub).is_dir():
                    stages.append({
                        "id": d.name,
                        "frames_path": f"{d.name}/{sub}",
                        "label": self._STAGE_LABELS.get(d.name, d.name),
                    })
                    break
        return stages

    def _list_frames(self, run_name: str, frames_rel: str):
        target = runs_dir() / run_name / frames_rel
        if not target.is_dir():
            return []
        return sorted([p.name for p in target.glob("*.png")])


def main():
    requested = int(os.environ.get("SILENT_FILM_VIEWER_PORT", "8765"))
    port = find_port(requested)
    handler = Handler
    with socketserver.TCPServer(("127.0.0.1", port), handler) as httpd:
        print(f"Silent Film viewer: http://127.0.0.1:{port}/")
        if port != requested:
            print(f"  (port {requested} was taken; bound to {port} instead)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("Shutting down viewer.")


if __name__ == "__main__":
    main()
