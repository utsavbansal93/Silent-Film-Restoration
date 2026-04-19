"""Structured logging + JOURNAL.md appender."""
from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from pipeline.common.paths import project_root

_CONFIGURED = False


def get_logger(name: str, run_dir: Path | None = None) -> logging.Logger:
    global _CONFIGURED
    logger = logging.getLogger(name)
    if _CONFIGURED:
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # Avoid duplicate console handlers.
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        root.addHandler(console)

    if run_dir is not None:
        log_path = run_dir / "run.log"
        fh = logging.FileHandler(log_path)
        fh.setFormatter(fmt)
        root.addHandler(fh)

    _CONFIGURED = True
    return logger


def append_journal(
    title: str,
    decision: str,
    alternatives: list[str] | None = None,
    why: str = "",
    revisit_if: str = "",
) -> None:
    """Append a structured decision entry to JOURNAL.md.

    Use this for every non-trivial technical call made during a run:
    library choice, threshold pick, fixture boundaries, port selection, etc.
    """
    journal = project_root() / "JOURNAL.md"
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    lines = [
        "",
        f"### {ts} — {title}",
        "",
        f"**Decision:** {decision}",
    ]
    if alternatives:
        lines.append("")
        lines.append("**Alternatives considered:**")
        for a in alternatives:
            lines.append(f"- {a}")
    if why:
        lines.append("")
        lines.append(f"**Why:** {why}")
    if revisit_if:
        lines.append("")
        lines.append(f"**Revisit if:** {revisit_if}")

    with journal.open("a") as f:
        f.write("\n".join(lines) + "\n")
