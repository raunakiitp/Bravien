"""Safe filesystem paths (§53).

Any path that originates outside the process — a request body, a CLI argument, a
checkpoint name — goes through `safe_join` before it touches the disk. The check
is resolution-based rather than string-based: `..` is only one of many ways to
escape a directory, and symlinks defeat prefix matching on the raw string.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

#: Names Windows refuses regardless of extension.
_WINDOWS_RESERVED = frozenset(
    ["CON", "PRN", "AUX", "NUL"]
    + [f"COM{i}" for i in range(1, 10)]
    + [f"LPT{i}" for i in range(1, 10)]
)

_UNSAFE_CHARS = re.compile(r'[<>:"|?*\x00-\x1f]')


class UnsafePathError(ValueError):
    """A path escaped its permitted root, or could not be made safe."""


def project_root() -> Path:
    """Repository root, derived from this file's location."""
    return Path(__file__).resolve().parents[2]


def safe_join(root: str | Path, *parts: str) -> Path:
    """Join `parts` under `root`, refusing anything that escapes it.

    Both sides are fully resolved before comparison, so `..` segments, absolute
    paths, and symlinks pointing outside the root are all caught.

    Raises:
        UnsafePathError: if the result would fall outside `root`.
    """
    base = Path(root).resolve()
    if not parts:
        return base

    for part in parts:
        if part is None or "\x00" in str(part):
            raise UnsafePathError("path component contains a null byte")

    candidate = base.joinpath(*(str(p) for p in parts))

    # strict=False: the target need not exist yet (we may be about to create it),
    # but every existing component, including symlinks, is resolved.
    resolved = candidate.resolve(strict=False)

    if resolved != base and base not in resolved.parents:
        raise UnsafePathError(
            f"path {'/'.join(str(p) for p in parts)!r} resolves outside {base}"
        )
    return resolved


def sanitize_filename(name: str, *, fallback: str = "file", max_length: int = 120) -> str:
    """Reduce an untrusted string to a single safe filename component.

    Directory separators are stripped rather than escaped: this returns a *name*,
    never a path.
    """
    name = str(name).replace("\x00", "")
    name = os.path.basename(name.replace("\\", "/"))
    name = _UNSAFE_CHARS.sub("_", name).strip(" .")

    if not name:
        return fallback

    stem, dot, suffix = name.rpartition(".")
    if dot and stem.upper() in _WINDOWS_RESERVED:
        name = f"{stem}_{dot}{suffix}"
    elif not dot and name.upper() in _WINDOWS_RESERVED:
        name = f"{name}_"

    if len(name) > max_length:
        stem, dot, suffix = name.rpartition(".")
        if dot and len(suffix) <= 10:
            keep = max_length - len(suffix) - 1
            name = f"{stem[:keep]}.{suffix}"
        else:
            name = name[:max_length]

    return name or fallback


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} TB"
