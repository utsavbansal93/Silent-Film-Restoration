"""SHA256 helpers for frames and config sections."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_bytes(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode())


def sample_frame_indices(total: int, n: int = 5) -> list[int]:
    """Deterministic sample: evenly spaced, always includes first and last."""
    if total <= 0:
        return []
    if total <= n:
        return list(range(total))
    # Pick 0, last, and evenly spaced middle.
    if n == 1:
        return [0]
    step = (total - 1) / (n - 1)
    idxs = [int(round(i * step)) for i in range(n)]
    # Dedup while preserving order.
    seen: set[int] = set()
    out: list[int] = []
    for i in idxs:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def hash_sampled_frames(frame_paths: list[Path], n: int = 5) -> dict[str, str]:
    """Return {frame_filename: sha256} for n evenly-spaced sampled frames."""
    frame_paths = sorted(frame_paths)
    idxs = sample_frame_indices(len(frame_paths), n)
    return {frame_paths[i].name: sha256_file(frame_paths[i]) for i in idxs}


def hash_iter(chunks: Iterable[bytes]) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(c)
    return h.hexdigest()
