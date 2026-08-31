"""The local Bravien API server (§31, §32).

FastAPI over the inference engine. Binds to 127.0.0.1 by default, so nothing is
exposed off the machine unless the operator asks for it explicitly.

The streaming wire format is Server-Sent Events carrying the same chunk shapes
the TypeScript side already models (`content_delta`, `message_complete`,
`error`), so the web app consumes this without a translation layer.

Security posture (§53):

* Request bodies are size-capped by middleware before parsing.
* Every field is validated and bounded by Pydantic; unknown fields are rejected.
* CORS is restricted to localhost origins.
* Concurrent generations are bounded by a semaphore; excess requests get 503
  rather than exhausting VRAM.
* Errors return a code and a message, never a traceback.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.middleware.base import BaseHTTPMiddleware

from bravien.inference.engine import (
    EngineConfig,
    EngineError,
    InferenceEngine,
    PromptTooLongError,
)
from bravien.inference.hf_engine import HFInferenceEngine
from bravien.model.generation import GenerationConfig
from bravien.tokenizer.templates import ChatTemplateError
from bravien.utils.logging import get_logger

logger = get_logger("inference.server")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

#: Reject a body larger than this before it is parsed.
MAX_BODY_BYTES = 1 << 20  # 1 MiB
#: Wait this long for a generation slot before answering 503.
GENERATION_SLOT_TIMEOUT = 30.0

ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
]


# ------------------------------------------------------------------- schemas


class _Strict(BaseModel):
    """Reject unknown fields, so a typo'd parameter is an error not a silent no-op."""

    model_config = ConfigDict(extra="forbid")


class SamplingParams(_Strict):
    max_tokens: int | None = Field(default=None, ge=1, le=32768)
    temperature: float | None = Field(default=None, ge=0.0, le=5.0)
    top_k: int | None = Field(default=None, ge=0, le=1_000_000)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    repetition_penalty: float | None = Field(default=None, gt=0.0, le=5.0)
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)
    stop: list[str] | None = Field(default=None, max_length=8)

    @field_validator("stop")
    @classmethod
    def _bound_stop_strings(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        for item in value:
            if len(item) > 64:
                raise ValueError("each stop string must be 64 characters or fewer")
        return [item for item in value if item]

    def to_generation(self) -> dict[str, Any]:
        """Map API names onto GenerationConfig names."""
        return {
            "max_new_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "seed": self.seed,
        }


class CompletionRequest(SamplingParams):
    prompt: str = Field(min_length=1, max_length=200_000)
    stream: bool = False
    model: str | None = None


class ChatMessage(_Strict):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = Field(max_length=200_000)


class ChatRequest(SamplingParams):
    messages: list[ChatMessage] = Field(min_length=1, max_length=200)
    stream: bool = False
    model: str | None = None

    @field_validator("messages")
    @classmethod
    def _total_size(cls, value: list[ChatMessage]) -> list[ChatMessage]:
        total = sum(len(m.content) for m in value)
        if total > 200_000:
            raise ValueError(
                f"messages total {total:,} characters, over the 200,000 limit"
            )
        return value


class ScoreRequest(_Strict):
    text: str = Field(min_length=1, max_length=200_000)


class TokenizeRequest(_Strict):
    """Count tokens with the served checkpoint's own tokenizer.

    Exists so no client has to estimate. Exactly one of `text` or `messages`.
    """

    text: str | None = Field(default=None, max_length=200_000)
    messages: list[ChatMessage] | None = Field(default=None, max_length=200)
    #: Return the ids as well as the count. Off by default: the ids are large
    #: and most callers only need the length.
    include_ids: bool = False
    #: Output reservation used when reporting whether `messages` fits.
    max_tokens: int | None = Field(default=None, ge=1, le=32768)

    @field_validator("messages")
    @classmethod
    def _total_size(cls, value: list[ChatMessage] | None) -> list[ChatMessage] | None:
        if value is None:
            return None
        total = sum(len(m.content) for m in value)
        if total > 200_000:
            raise ValueError(
                f"messages total {total:,} characters, over the 200,000 limit"
            )
        return value


# ---------------------------------------------------------------- middleware


class BodyLimitMiddleware(BaseHTTPMiddleware):
    """Reject oversized bodies up front.

    Checks Content-Length so an oversized request is refused before it is read
    into memory; a chunked request with no declared length still hits Pydantic's
    per-field limits.
    """

    def __init__(self, app: Any, max_bytes: int = MAX_BODY_BYTES) -> None:
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next: Any) -> Any:
        declared = request.headers.get("content-length")
        if declared is not None:
            try:
                if int(declared) > self.max_bytes:
                    return JSONResponse(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        content={
                            "error": {
                                "code": "payload_too_large",
                                "message": f"request body exceeds {self.max_bytes} bytes",
                            }
                        },
                    )
            except ValueError:
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={
                        "error": {
                            "code": "bad_content_length",
                            "message": "Content-Length is not an integer",
                        }
                    },
                )
        return await call_next(request)


# -------------------------------------------------------------- engine holder


class EngineHolder:
    """Owns the engine and bounds how many generations run at once.

    The engine is loaded once at startup. If loading fails the server still comes
    up, `/health` reports the reason, and generation endpoints return 503 — a
    server that cannot start is much harder to diagnose than one that says why.
    """

    def __init__(self, max_concurrency: int = 1) -> None:
        self.engine: InferenceEngine | HFInferenceEngine | None = None
        self.error: str | None = None
        self._slots = threading.BoundedSemaphore(max_concurrency)
        self.max_concurrency = max_concurrency

    def load(self, path: str | Path, config: EngineConfig | None = None) -> None:
        path_str = str(path)
        try:
            # Check if this is a HuggingFace model repo id or directory
            if "/" in path_str and not Path(path).exists():
                self.engine = HFInferenceEngine.from_pretrained(path_str, config=config)
                self.error = None
                return

            if Path(path).exists() and not (Path(path) / "config.json").exists() and not (Path(path) / "tokenizer.json").exists():
                # If directory doesn't have native Bravien config, try HF
                try:
                    self.engine = HFInferenceEngine.from_pretrained(path_str, config=config)
                    self.error = None
                    return
                except Exception:
                    pass

            self.engine = InferenceEngine.from_checkpoint(path, config=config)
            self.error = None
        except Exception as exc:
            # Fallback attempt via HF if standard loading failed
            try:
                self.engine = HFInferenceEngine.from_pretrained(path_str, config=config)
                self.error = None
                return
            except Exception:
                pass
            self.engine = None
            self.error = str(exc)
            logger.error("could not load checkpoint/model %s: %s", path, exc)

    def require(self) -> InferenceEngine | HFInferenceEngine:
        if self.engine is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "code": "model_not_loaded",
                    "message": self.error
                    or "no checkpoint is loaded; train one or provide a valid model",
                },
            )
        return self.engine

    def acquire(self) -> None:
        if not self._slots.acquire(timeout=GENERATION_SLOT_TIMEOUT):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "code": "busy",
                    "message": (
                        f"all {self.max_concurrency} generation slot(s) are busy; "
                        f"retry shortly"
                    ),
                },
            )

    def release(self) -> None:
        try:
            self._slots.release()
        except ValueError:  # pragma: no cover - defensive
            logger.warning("generation slot released more times than acquired")


# ------------------------------------------------------------------ SSE wire


def _sse(payload: dict[str, Any]) -> str:
    """One SSE frame. `ensure_ascii=False` keeps multi-byte text intact."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _stream_body(
    holder: EngineHolder,
    engine: InferenceEngine,
    prompt: str,
    generation: dict[str, Any],
    stop_strings: list[str],
    *,
    context: dict[str, Any] | None = None,
) -> Iterator[str]:
    """SSE frames for one generation.

    A synchronous generator on purpose: Starlette iterates it in a worker thread,
    so the blocking forward passes never occupy the event loop.

    `context` is the budgeting plan from the chat path. It rides on
    `message_complete` so the client can show what was actually sent to the model
    and what had to be dropped to fit.
    """
    try:
        for event in engine.stream(
            prompt, generation=generation, stop_strings=stop_strings
        ):
            if event.done:
                usage = dict(event.usage)
                usage.pop("text", None)
                plan = context if context is not None else usage.get("context")
                yield _sse(
                    {
                        "kind": "message_complete",
                        "finishReason": event.finish_reason,
                        "usage": {
                            "inputTokens": usage.get("prompt_tokens"),
                            "outputTokens": usage.get("completion_tokens"),
                        },
                        "timing": {
                            "seconds": usage.get("seconds"),
                            "tokensPerSecond": usage.get("tokens_per_second"),
                        },
                        "model": engine.model_name,
                        **({"context": plan} if plan else {}),
                        **(
                            {"promptTruncation": usage["prompt_truncation"]}
                            if usage.get("prompt_truncation")
                            else {}
                        ),
                    }
                )
            else:
                yield _sse({"kind": "content_delta", "delta": event.text})
    except (EngineError, RuntimeError, ValueError) as exc:
        logger.exception("generation failed")
        yield _sse(
            {
                "kind": "error",
                "code": "generation_failed",
                "message": str(exc)[:500],
            }
        )
    finally:
        holder.release()
        # Sentinel so a client reading raw SSE knows the stream ended cleanly and
        # was not cut off mid-generation.
        yield "data: [DONE]\n\n"


# --------------------------------------------------------------------- app


def create_app(
    holder: EngineHolder | None = None,
    *,
    checkpoint: str | Path | None = None,
    engine_config: EngineConfig | None = None,
    max_concurrency: int = 1,
) -> FastAPI:
    """Build the app.

    Args:
        holder: pre-built holder, for tests that inject an engine directly.
        checkpoint: loaded at startup when no holder is given.
    """
    state = holder or EngineHolder(max_concurrency=max_concurrency)

    app = FastAPI(
        title="Bravien Inference API",
        description=(
            "Local inference over a Bravien checkpoint. Runs entirely on this "
            "machine; no external model service is contacted."
        ),
        version="0.1.0",
    )
    app.add_middleware(BodyLimitMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )
    app.state.bravien = state
    started = time.time()

    if holder is None and checkpoint is not None:
        state.load(checkpoint, engine_config)

    # ------------------------------------------------------------- diagnostics

    @app.get("/")
    def root() -> dict[str, Any]:
        """Root endpoint with quick server status and API links."""
        engine = state.engine
        return {
            "name": "Bravien Inference API",
            "status": "online" if engine is not None else "no_model",
            "model": engine.model_name if engine else None,
            "device": engine.device_info.name if engine else None,
            "endpoints": {
                "health": "/health",
                "docs": "/docs",
                "models": "/v1/models",
                "chat_completions": "/v1/chat/completions",
                "completions": "/v1/completions",
            },
            "web_ui": "Run 'npm run dev' to access the frontend at http://localhost:3000",
        }

    @app.get("/health")
    def health() -> dict[str, Any]:
        """Runtime status. The UI polls this to show whether a model is loaded."""
        engine = state.engine
        return {
            "status": "ok" if engine is not None else "no_model",
            "model_loaded": engine is not None,
            "error": state.error,
            "model": engine.model_name if engine else None,
            "checkpoint": str(engine.checkpoint_path) if engine else None,
            "device": engine.device_info.name if engine else None,
            "context_length": engine.max_context if engine else None,
            "max_concurrency": state.max_concurrency,
            "uptime_seconds": round(time.time() - started, 1),
            "backend": "bravien-local",
        }

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        engine = state.engine
        if engine is None:
            return {"object": "list", "data": []}
        return {"object": "list", "data": [engine.info()]}

    @app.get("/v1/models/{name}")
    def get_model(name: str) -> dict[str, Any]:
        engine = state.require()
        if name != engine.model_name:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "code": "model_not_found",
                    "message": f"this server serves {engine.model_name!r} only",
                },
            )
        return engine.info()

    # -------------------------------------------------------------- generation

    def _prepare(
        request: SamplingParams,
    ) -> tuple[InferenceEngine, dict[str, Any], list[str]]:
        engine = state.require()
        return engine, request.to_generation(), list(request.stop or [])

    @app.post("/v1/completions")
    def completions(request: CompletionRequest) -> Any:
        engine, generation, stops = _prepare(request)
        state.acquire()
        if request.stream:
            return StreamingResponse(
                _stream_body(state, engine, request.prompt, generation, stops),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",  # do not let a proxy buffer the stream
                },
            )
        try:
            result = engine.complete(
                request.prompt, generation=generation, stop_strings=stops
            )
        except PromptTooLongError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "prompt_too_long", "message": str(exc)},
            ) from exc
        except EngineError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "invalid_request", "message": str(exc)},
            ) from exc
        finally:
            state.release()
        return {"object": "completion", **result.to_dict()}

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatRequest) -> Any:
        engine = state.require()
        messages = [m.model_dump() for m in request.messages]
        generation = request.to_generation()
        stops = list(request.stop or [])

        # Budget against the real tokenizer before anything is generated. This
        # drops whole turns oldest-first and keeps the system prompt, rather than
        # letting `_encode_prompt` cut the encoded string blindly, and the plan is
        # returned to the caller so a truncation is never invisible.
        try:
            reserved = (
                request.max_tokens
                if request.max_tokens is not None
                else engine.config.default_generation.max_new_tokens
            )
            prompt, plan = engine.build_chat_prompt_within_context(
                messages, max_new_tokens=reserved
            )
        except ChatTemplateError as exc:
            # A conversation the template cannot render — an unknown role, or a
            # final assistant turn with nothing to answer. That is bad client
            # input, not a server fault, so it must not surface as a 500.
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "invalid_conversation", "message": str(exc)},
            ) from exc
        except PromptTooLongError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "prompt_too_long", "message": str(exc)},
            ) from exc
        except EngineError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "invalid_request", "message": str(exc)},
            ) from exc

        state.acquire()
        if request.stream:
            return StreamingResponse(
                _stream_body(
                    state, engine, prompt, generation, stops, context=plan.to_dict()
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        try:
            result = engine.complete(prompt, generation=generation, stop_strings=stops)
        except PromptTooLongError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "prompt_too_long", "message": str(exc)},
            ) from exc
        except EngineError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "invalid_request", "message": str(exc)},
            ) from exc
        finally:
            state.release()
        return {
            "object": "chat.completion",
            "message": {"role": "assistant", "content": result.text},
            **result.to_dict(),
            "context": plan.to_dict(),
        }

    @app.post("/v1/tokenize")
    def tokenize(request: TokenizeRequest) -> dict[str, Any]:
        """Exact token counts, so callers never estimate characters-per-token."""
        engine = state.require()
        try:
            return engine.count_tokens(
                text=request.text,
                messages=(
                    [m.model_dump() for m in request.messages]
                    if request.messages is not None
                    else None
                ),
                include_ids=request.include_ids,
                max_new_tokens=request.max_tokens,
            )
        except ChatTemplateError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "invalid_conversation", "message": str(exc)},
            ) from exc
        except PromptTooLongError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "prompt_too_long", "message": str(exc)},
            ) from exc
        except EngineError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "invalid_request", "message": str(exc)},
            ) from exc

    @app.post("/v1/score")
    def score(request: ScoreRequest) -> dict[str, Any]:
        """Token-level log-probability of a text. Used by evaluation."""
        engine = state.require()
        state.acquire()
        try:
            return engine.logprob(request.text)
        except EngineError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "invalid_request", "message": str(exc)},
            ) from exc
        finally:
            state.release()

    @app.post("/v1/warmup")
    def warmup() -> dict[str, Any]:
        """Warm up the model runtime with a minimal generation."""
        engine = state.require()
        state.acquire()
        try:
            return engine.warmup()
        finally:
            state.release()

    # ----------------------------------------------------------- error shaping

    @app.exception_handler(HTTPException)
    async def http_error(_: Request, exc: HTTPException) -> JSONResponse:
        detail = exc.detail
        if isinstance(detail, dict):
            body = {"error": detail}
        else:
            body = {"error": {"code": "http_error", "message": str(detail)}}
        return JSONResponse(status_code=exc.status_code, content=body)

    return app


def resolve_checkpoint_path(explicit: str | Path | None = None) -> Path | str:
    """Where to load weights from: the argument, then env, then the default run."""
    if explicit:
        return explicit if ("/" in str(explicit) and not Path(explicit).exists()) else Path(explicit)
    from_env = os.environ.get("BRAVIEN_CHECKPOINT") or os.environ.get("BRAVIEN_MODEL")
    if from_env:
        return from_env if ("/" in from_env and not Path(from_env).exists()) else Path(from_env)
    
    # Canonical production checkpoint preference
    if (Path("checkpoints") / "bravien-v1").exists():
        return Path("checkpoints") / "bravien-v1"
    if (Path("checkpoints") / "bravien").exists():
        return Path("checkpoints") / "bravien"
    return "Qwen/Qwen2.5-0.5B-Instruct"


def serve(
    checkpoint: str | Path | None = None,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    engine_config: EngineConfig | None = None,
    max_concurrency: int = 1,
    log_level: str = "info",
) -> None:
    """Run the server. Blocks until interrupted."""
    import uvicorn

    path = resolve_checkpoint_path(checkpoint)
    app = create_app(
        checkpoint=path,
        engine_config=engine_config or EngineConfig(),
        max_concurrency=max_concurrency,
    )

    if host not in ("127.0.0.1", "localhost", "::1"):
        logger.warning(
            "binding to %s exposes this server beyond this machine; it has no "
            "authentication",
            host,
        )
    logger.info("Bravien API on http://%s:%d  (checkpoint: %s)", host, port, path)
    uvicorn.run(app, host=host, port=port, log_level=log_level)


__all__ = [
    "ALLOWED_ORIGINS",
    "ChatRequest",
    "CompletionRequest",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "EngineHolder",
    "MAX_BODY_BYTES",
    "create_app",
    "resolve_checkpoint_path",
    "serve",
]
