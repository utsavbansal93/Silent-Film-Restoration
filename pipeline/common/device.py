"""Runtime device detection. Pass 1 doesn't use GPU, but stages must still log device."""
from __future__ import annotations

import platform


def detect_device() -> str:
    try:
        import torch  # optional: only if torch is installed

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def device_label() -> str:
    d = detect_device()
    if d == "mps":
        return "mps/apple_m_series"
    if d == "cuda":
        try:
            import torch

            return f"cuda/{torch.cuda.get_device_name(0).lower().replace(' ', '_')}"
        except Exception:
            return "cuda/unknown"
    return f"cpu/{platform.machine()}"
