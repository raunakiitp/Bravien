"""Console and file logging (§53, §62).

Two jobs beyond ordinary logging:

* **UTF-8 output.** Windows consoles default to a legacy code page, so printing a
  tokenizer sample containing CJK or emoji raises `UnicodeEncodeError` and kills
  the run. Every Bravien entry point calls `configure_stdout()` first.
* **Safe logging.** Log lines must never carry an API key or a bearer token, even
  when a caller passes one in by accident (§53).
"""

from __future__ import annotations

import logging
import os
import re
import sys
import time
from pathlib import Path

#: Patterns redacted from every log record. Ordered longest-context first so a
#: `key=value` form is caught before the bare value.
_REDACTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?i)\b(api[_-]?key|secret|password|passwd|token|authorization)\b"
                r"\s*[:=]\s*['\"]?([^\s'\"&,]+)"), r"\1=<redacted>"),
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{8,}"), "Bearer <redacted>"),
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}\b"), "<redacted>"),
    (re.compile(r"\b(?:gh[pousr]|github_pat)_[A-Za-z0-9_]{16,}\b"), "<redacted>"),
    (re.compile(r"\bAIza[0-9A-Za-z_\-]{20,}\b"), "<redacted>"),
    (re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b"), "<redacted>"),
    (re.compile(r"postgres(?:ql)?://[^:@\s]+:[^@\s]+@"), "postgresql://<redacted>@"),
)


def redact(text: str) -> str:
    """Strip credential-shaped substrings from a string."""
    for pattern, replacement in _REDACTIONS:
        text = pattern.sub(replacement, text)
    return text


class RedactingFormatter(logging.Formatter):
    """Formatter that redacts secrets from the fully rendered line.

    Redacting after formatting is deliberate: it covers the message, its
    arguments and any exception text in one pass, so a secret cannot slip
    through via `%s` interpolation.
    """

    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))


def configure_stdout() -> None:
    """Force UTF-8 on stdout/stderr.

    Without this, `print()` of any non-latin-1 text crashes on a default Windows
    console (cp1252). Called by every script entry point before doing work.
    """
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            # errors="replace": a console that genuinely cannot render a glyph
            # should print a placeholder, not abort the training run.
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


_CONFIGURED = False


def setup_logging(
    level: str | int = "INFO",
    *,
    log_file: str | Path | None = None,
    quiet: bool = False,
) -> logging.Logger:
    """Configure the `bravien` logger tree. Idempotent.

    Args:
        level: threshold name or value; `BRAVIEN_LOG_LEVEL` overrides it.
        log_file: optional file to receive the same records.
        quiet: suppress console output (file logging still applies).
    """
    global _CONFIGURED
    configure_stdout()

    level = os.environ.get("BRAVIEN_LOG_LEVEL", level)
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    logger = logging.getLogger("bravien")
    logger.setLevel(level)

    if _CONFIGURED:
        return logger

    # Own handlers only: never touch the root logger, so importing Bravien from
    # another application does not hijack its logging.
    logger.propagate = False
    fmt = RedactingFormatter(
        "%(asctime)s %(levelname)-7s %(name)s  %(message)s", datefmt="%H:%M:%S"
    )

    if not quiet:
        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(fmt)
        logger.addHandler(console)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    _CONFIGURED = True
    return logger


def get_logger(name: str = "bravien") -> logging.Logger:
    if not name.startswith("bravien"):
        name = f"bravien.{name}"
    return logging.getLogger(name)


def format_duration(seconds: float) -> str:
    """Human-readable elapsed time, for progress lines."""
    seconds = max(seconds, 0.0)
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def format_count(n: float) -> str:
    """Compact counts: 1234 -> 1.23K, 4.5e9 -> 4.50B."""
    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(n) >= threshold:
            return f"{n / threshold:.2f}{suffix}"
    return f"{n:.0f}"


class Timer:
    """Context manager measuring a block, used for training-step timings."""

    def __init__(self) -> None:
        self.elapsed = 0.0
        self._start = 0.0

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed = time.perf_counter() - self._start

    def __str__(self) -> str:
        return format_duration(self.elapsed)
