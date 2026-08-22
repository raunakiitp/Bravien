"""Bravien inference: the engine and the local API server (§25, §29–§32).

`InferenceEngine` is the only component that owns loaded weights. Everything
above it — the API, the web app — talks to a Bravien checkpoint through this
layer and nowhere else.
"""

from __future__ import annotations

from bravien.inference.engine import (
    MAX_PROMPT_CHARS,
    EngineConfig,
    EngineError,
    GenerationResult,
    InferenceEngine,
    PromptTooLongError,
    StreamDecoder,
    StreamEvent,
)

__all__ = [
    "EngineConfig",
    "EngineError",
    "GenerationResult",
    "InferenceEngine",
    "MAX_PROMPT_CHARS",
    "PromptTooLongError",
    "StreamDecoder",
    "StreamEvent",
]


def __getattr__(name: str) -> object:
    """Load the server lazily.

    `bravien.inference` must be importable for training and evaluation on a
    machine with no FastAPI installed, so the server is only imported when it is
    actually asked for.
    """
    if name in ("create_app", "serve", "EngineHolder"):
        from bravien.inference import server

        return getattr(server, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
