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
        return sorted([d.name for d in runs_dir().iterdir() if d.is_dir()], reverse=True)

    def _list_stages(self, run_name: str):
        run = runs_dir() / run_name
        if not run.is_dir():
            return []
        stages = []
        for d in sorted(run.iterdir()):
            if not d.is_dir():
                continue
            # Prefer the frames subdir for display.
            for sub in ("frames_raw", "frames_stabilised"):
                if (d / sub).is_dir():
                    stages.append({"id": d.name, "frames_path": f"{d.name}/{sub}"})
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
