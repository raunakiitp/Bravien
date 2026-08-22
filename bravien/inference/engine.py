"""The Bravien inference engine (§25, §29–§32).

This is the only place that owns a loaded checkpoint. It holds the weights and
the tokenizer together, applies the chat template, decides where generation
stops, and turns token ids back into text incrementally.

Two things here are less obvious than they look:

* **Incremental detokenization.** A byte-level BPE token can be a fragment of a
  multi-byte character, so decoding tokens one at a time and concatenating
  produces mojibake. `StreamDecoder` decodes the whole sequence each step and
  emits only the new suffix, holding back anything that ends mid-character.
* **Stop tokens.** The model layer has no tokenizer, so `GenerationConfig`
  ships with none. The engine fills them in from the tokenizer's EOS and
  `</ASSISTANT>` ids — without that, chat generation runs to `max_new_tokens`
  every time.

No network call happens anywhere in this file (§2).
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from bravien.model.generation import GenerationConfig, generate
from bravien.model.model import BravienForCausalLM
from bravien.tokenizer.special_tokens import ASSISTANT_CLOSE, EOS
from bravien.tokenizer.tokenizer import BravienTokenizer
from bravien.training.checkpoint import (
    CheckpointMetadata,
    load_checkpoint,
    resolve_latest,
)
from bravien.utils.hardware import DeviceInfo, detect_device
from bravien.utils.logging import get_logger

logger = get_logger("inference.engine")

#: Refuse a prompt longer than this many characters before tokenizing it, so a
#: huge request cannot be turned into a huge allocation (§53).
MAX_PROMPT_CHARS = 200_000


class EngineError(RuntimeError):
    pass


class PromptTooLongError(EngineError):
    """The prompt does not leave room to generate inside the context window."""


class StreamDecoder:
    """Incremental token-to-text decoding for byte-level BPE.

    Decodes the accumulated sequence on each push and returns the delta. The
    quadratic decode cost is irrelevant next to a forward pass, and it is the
    only approach that keeps multi-byte characters and BPE whitespace markers
    intact.
    """

    def __init__(self, tokenizer: BravienTokenizer, *, skip_special: bool = True) -> None:
        self._tokenizer = tokenizer
        self._skip_special = skip_special
        self._ids: list[int] = []
        self._emitted = 0
        self.text = ""

    def push(self, token_id: int) -> str:
        """Add one token and return the newly decodable text, possibly empty."""
        self._ids.append(token_id)
        decoded = self._tokenizer.decode(
            self._ids, skip_special_tokens=self._skip_special
        )

        # A trailing U+FFFD means the last token was a partial UTF-8 sequence.
        # Hold it: the next token completes the character.
        if decoded.endswith("�"):
            return ""

        delta = decoded[self._emitted :]
        self._emitted = len(decoded)
        self.text = decoded
        return delta

    def flush(self) -> str:
        """Emit whatever was held back, replacement characters and all."""
        decoded = self._tokenizer.decode(
            self._ids, skip_special_tokens=self._skip_special
        )
        delta = decoded[self._emitted :]
        self._emitted = len(decoded)
        self.text = decoded
        return delta


@dataclass
class GenerationResult:
    """A finished completion, with measured counts only."""

    text: str
    token_ids: list[int]
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    seconds: float
    model: str = ""

    @property
    def tokens_per_second(self) -> float:
        return self.completion_tokens / self.seconds if self.seconds > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "finish_reason": self.finish_reason,
            "seconds": round(self.seconds, 4),
            "tokens_per_second": round(self.tokens_per_second, 2),
            "model": self.model,
        }


@dataclass
class StreamEvent:
    """One streaming step.

    `text` is a delta, never cumulative. `done` arrives exactly once, last.
    """

    text: str = ""
    token_id: int | None = None
    index: int = 0
    done: bool = False
    finish_reason: str = ""
    usage: dict[str, Any] = field(default_factory=dict)


@dataclass
class EngineConfig:
    checkpoint: Path | None = None
    device: str | None = None
    #: "auto" follows the hardware; fp32 is the safe fallback everywhere.
    dtype: str = "auto"
    #: Applied when a request does not specify one.
    default_generation: GenerationConfig = field(default_factory=GenerationConfig)
    #: A single hard cap on output length, whatever a request asks for.
    max_new_tokens_limit: int = 2048

    def __post_init__(self) -> None:
        if self.checkpoint is not None:
            self.checkpoint = Path(self.checkpoint)
        if self.dtype not in ("auto", "bf16", "fp16", "fp32"):
            raise ValueError(f"unknown dtype {self.dtype!r}")


def _held_back_for_stop_strings(text: str, stop_strings: Sequence[str]) -> int:
    """Length of the trailing text that could still turn into a stop string.

    A delta the caller has already received cannot be withdrawn, so emitting
    `"STO"` and only then discovering `"STOP"` leaks past the caller's own stop
    sequence. Any suffix that is a proper prefix of a stop string therefore waits
    for the next token to resolve it, and the final flush releases it if no stop
    string ever materialised.
    """
    longest = 0
    for stop in stop_strings:
        for n in range(min(len(stop) - 1, len(text)), longest, -1):
            if text.endswith(stop[:n]):
                longest = n
                break
    return longest


def _resolve_dtype(requested: str, device: DeviceInfo) -> torch.dtype:
    """Inference dtype.

    fp32 on CPU unconditionally: half precision on CPU is emulated and slower,
    not faster.
    """
    if requested == "fp32" or not device.is_cuda:
        return torch.float32
    if requested == "bf16":
        return torch.bfloat16
    if requested == "fp16":
        return torch.float16
    return torch.bfloat16 if device.supports_bf16 else torch.float16


class InferenceEngine:
    """Loads a Bravien checkpoint and generates from it.

    Constructed via `from_checkpoint`, which is the only supported entry point:
    a model without its matching tokenizer is not a usable engine.
    """

    def __init__(
        self,
        model: BravienForCausalLM,
        tokenizer: BravienTokenizer,
        *,
        config: EngineConfig | None = None,
        metadata: CheckpointMetadata | None = None,
        device: DeviceInfo | None = None,
        checkpoint_path: Path | None = None,
    ) -> None:
        self.config = config or EngineConfig()
        self.device_info = device or detect_device(self.config.device)
        self.device = torch.device(self.device_info.device)
        self.dtype = _resolve_dtype(self.config.dtype, self.device_info)

        self.tokenizer = tokenizer
        self.checkpoint_metadata = metadata or CheckpointMetadata()
        self.checkpoint_path = checkpoint_path

        if tokenizer.vocab_size != model.config.vocab_size:
            raise EngineError(
                f"tokenizer vocab ({tokenizer.vocab_size}) does not match the "
                f"model ({model.config.vocab_size})"
            )

        self.model = model.to(device=self.device, dtype=self.dtype).eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.stop_token_ids = self._collect_stop_ids()
        self.loaded_at = time.time()
        self._generations = 0
        self._completion_tokens = 0

        logger.info(
            "engine ready: %s, %s params, %s on %s, stop ids %s",
            self.model_name,
            f"{self.model.parameter_report().total:,}",
            str(self.dtype).replace("torch.", ""),
            self.device_info.name,
            sorted(self.stop_token_ids),
        )

    # ------------------------------------------------------------- construction

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        config: EngineConfig | None = None,
    ) -> InferenceEngine:
        """Load a checkpoint directory, or the newest one inside a run directory."""
        path = Path(path)
        if not path.exists():
            raise EngineError(
                f"no checkpoint at {path}. Train one first, or point "
                f"BRAVIEN_CHECKPOINT at an existing checkpoint directory."
            )

        # A run directory holds step-* subdirectories; a checkpoint holds config.json.
        if not (path / "config.json").exists():
            latest = resolve_latest(path)
            if latest is None:
                raise EngineError(
                    f"{path} is neither a checkpoint nor a run directory "
                    f"containing one"
                )
            logger.info("resolved %s to %s", path, latest.name)
            path = latest

        engine_config = config or EngineConfig()
        engine_config.checkpoint = path
        device = detect_device(engine_config.device)

        loaded = load_checkpoint(
            path,
            device="cpu",  # move once, after casting, in __init__
            load_tokenizer=True,
            load_training_state=False,
        )
        if loaded.tokenizer is None:
            raise EngineError(
                f"{path} has no bundled tokenizer. A checkpoint must carry the "
                f"tokenizer it was trained with, or its token ids are unreadable."
            )

        return cls(
            loaded.model,
            loaded.tokenizer,
            config=engine_config,
            metadata=loaded.metadata,
            device=device,
            checkpoint_path=path,
        )

    def _collect_stop_ids(self) -> tuple[int, ...]:
        """EOS plus the assistant close marker, whichever exist."""
        ids: list[int] = []
        for token in (EOS, ASSISTANT_CLOSE):
            token_id = self.tokenizer.token_to_id(token)
            if token_id is not None:
                ids.append(token_id)
        if not ids:
            logger.warning(
                "tokenizer defines neither %s nor %s; generation can only stop "
                "at max_new_tokens",
                EOS,
                ASSISTANT_CLOSE,
            )
        return tuple(dict.fromkeys(ids))

    # ----------------------------------------------------------------- metadata

    @property
    def model_name(self) -> str:
        return self.model.config.name

    @property
    def max_context(self) -> int:
        return self.model.config.max_position_embeddings

    def info(self) -> dict[str, Any]:
        """What the model actually is — read from the checkpoint, not asserted.

        Everything here is either a config value or a measurement. The API and UI
        display this rather than any hardcoded description (§29, §71).
        """
        report = self.model.parameter_report()
        meta = self.checkpoint_metadata
        return {
            "name": self.model_name,
            "version": self.model.config.version,
            "architecture": "decoder-only transformer",
            "parameters": report.total,
            "layers": self.model.config.num_layers,
            "hidden_size": self.model.config.hidden_size,
            "heads": self.model.config.num_heads,
            "kv_heads": self.model.config.num_kv_heads,
            "context_length": self.max_context,
            "vocab_size": self.model.config.vocab_size,
            "norm": self.model.config.norm_kind,
            "position_encoding": self.model.config.position_encoding,
            "activation": self.model.config.activation,
            "precision": str(self.dtype).replace("torch.", ""),
            "device": self.device_info.name,
            "checkpoint": str(self.checkpoint_path) if self.checkpoint_path else None,
            "training": {
                "stage": meta.stage,
                "step": meta.step,
                "tokens_seen": meta.tokens_seen,
                "run_name": meta.run_name,
                "created_at": meta.created_at,
                "metrics": meta.metrics,
                "dataset": meta.dataset,
            },
            "tokenizer": {
                "vocab_size": self.tokenizer.vocab_size,
                "checksum": self.tokenizer.metadata.get("vocab_checksum"),
            },
            "served_generations": self._generations,
            "served_completion_tokens": self._completion_tokens,
            "uptime_seconds": round(time.time() - self.loaded_at, 1),
        }

    # --------------------------------------------------------------- generation

    def _merge_config(self, overrides: dict[str, Any] | None) -> GenerationConfig:
        """Build a request's config from the defaults plus its overrides.

        Stop ids always come from the engine, and `max_new_tokens` is always
        clamped, so neither can be widened by a request. `None` means "not
        specified" and falls back to the default; `top_k=0` / `top_p=1.0` are the
        conventional ways to switch those filters off.
        """
        base = self.config.default_generation
        values: dict[str, Any] = {
            "max_new_tokens": base.max_new_tokens,
            "temperature": base.temperature,
            "top_k": base.top_k,
            "top_p": base.top_p,
            "repetition_penalty": base.repetition_penalty,
            "do_sample": base.do_sample,
            "seed": base.seed,
        }
        if overrides:
            unknown = set(overrides) - set(values)
            if unknown:
                raise EngineError(f"unknown generation parameter(s): {sorted(unknown)}")
            supplied = {k: v for k, v in overrides.items() if v is not None}
            if supplied.get("top_k") == 0:
                supplied["top_k"] = None
            if supplied.get("top_p") is not None and supplied["top_p"] >= 1.0:
                supplied["top_p"] = None
            values.update(supplied)

        values["max_new_tokens"] = max(
            1, min(int(values["max_new_tokens"]), self.config.max_new_tokens_limit)
        )
        values["stop_token_ids"] = self.stop_token_ids
        return GenerationConfig(**values)

    def _encode_prompt(self, prompt: str, max_new_tokens: int) -> torch.Tensor:
        if not isinstance(prompt, str):
            raise EngineError("prompt must be a string")
        if len(prompt) > MAX_PROMPT_CHARS:
            raise PromptTooLongError(
                f"prompt is {len(prompt):,} characters, over the "
                f"{MAX_PROMPT_CHARS:,} character limit"
            )

        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if not ids:
            raise EngineError("prompt encoded to zero tokens")

        room = self.max_context - max_new_tokens
        if room <= 0:
            raise PromptTooLongError(
                f"max_new_tokens={max_new_tokens} fills the entire "
                f"{self.max_context}-token context, leaving no room for a prompt"
            )
        if len(ids) > room:
            # Keep the tail: in a conversation the most recent turns matter most,
            # and truncating the front is visible to the caller in the response.
            logger.warning(
                "prompt of %d tokens truncated to the last %d", len(ids), room
            )
            ids = ids[-room:]

        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def stream(
        self,
        prompt: str,
        *,
        generation: dict[str, Any] | None = None,
        stop_strings: Sequence[str] = (),
    ) -> Iterator[StreamEvent]:
        """Stream a completion for a raw prompt string.

        Every event's `text` is a real delta produced by the model on this call.
        Nothing is buffered to fake a smooth cadence (§52). The one thing that is
        deliberately delayed is a partial match against `stop_strings`, which
        cannot be emitted before it is known not to be a stop string.
        """
        config = self._merge_config(generation)
        input_ids = self._encode_prompt(prompt, config.max_new_tokens)
        prompt_tokens = int(input_ids.size(1))

        decoder = StreamDecoder(self.tokenizer)
        produced: list[int] = []
        finish_reason = "length"
        final_text: str | None = None
        # What the caller has seen, as an offset into the decoded text. The
        # decoder's own counter tracks partial characters; this one additionally
        # accounts for text withheld pending a possible stop string.
        emitted = 0
        started = time.perf_counter()

        for index, token_id in enumerate(generate(self.model, input_ids, config)):
            produced.append(token_id)

            if token_id in config.stop_token_ids:
                finish_reason = "stop"
                break

            decoder.push(token_id)
            text = decoder.text

            if stop_strings:
                hit = min(
                    (text.index(s) for s in stop_strings if s and s in text),
                    default=-1,
                )
                if hit >= 0:
                    # Emit only what precedes the stop string, then stop.
                    if hit > emitted:
                        yield StreamEvent(
                            text=text[emitted:hit], token_id=token_id, index=index
                        )
                    finish_reason = "stop_string"
                    final_text = text[:hit]
                    break
                safe = len(text) - _held_back_for_stop_strings(text, stop_strings)
            else:
                safe = len(text)

            if safe > emitted:
                yield StreamEvent(
                    text=text[emitted:safe], token_id=token_id, index=index
                )
                emitted = safe

        if final_text is None:
            decoder.flush()
            if len(decoder.text) > emitted:
                yield StreamEvent(text=decoder.text[emitted:], index=len(produced))
            final_text = decoder.text

        elapsed = time.perf_counter() - started
        self._generations += 1
        self._completion_tokens += len(produced)

        yield StreamEvent(
            done=True,
            finish_reason=finish_reason,
            index=len(produced),
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": len(produced),
                "total_tokens": prompt_tokens + len(produced),
                "seconds": round(elapsed, 4),
                "tokens_per_second": round(
                    len(produced) / elapsed if elapsed > 0 else 0.0, 2
                ),
                "text": final_text,
                "token_ids": list(produced),
            },
        )

    def complete(
        self,
        prompt: str,
        *,
        generation: dict[str, Any] | None = None,
        stop_strings: Sequence[str] = (),
    ) -> GenerationResult:
        """Non-streaming completion. Drains `stream`, so the two cannot diverge."""
        started = time.perf_counter()
        ids: list[int] = []
        usage: dict[str, Any] = {}
        finish_reason = "length"

        for event in self.stream(
            prompt, generation=generation, stop_strings=stop_strings
        ):
            if event.done:
                usage = event.usage
                finish_reason = event.finish_reason
            elif event.token_id is not None:
                ids.append(event.token_id)

        return GenerationResult(
            text=usage.get("text", ""),
            token_ids=list(usage.get("token_ids", ids)),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            finish_reason=finish_reason,
            seconds=time.perf_counter() - started,
            model=self.model_name,
        )

    # --------------------------------------------------------------------- chat

    def build_chat_prompt(self, messages: Sequence[dict[str, Any]]) -> str:
        """Render messages through the trained chat template.

        Content is sanitised by the template layer, so a user message containing
        `<ASSISTANT>` cannot forge a turn boundary.
        """
        if not messages:
            raise EngineError("messages must not be empty")
        for message in messages:
            if not isinstance(message, dict):
                raise EngineError("each message must be an object")
            if "role" not in message or "content" not in message:
                raise EngineError("each message needs a role and content")
        return self.tokenizer.apply_chat_template(messages)

    def stream_chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        generation: dict[str, Any] | None = None,
    ) -> Iterator[StreamEvent]:
        return self.stream(self.build_chat_prompt(messages), generation=generation)

    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        generation: dict[str, Any] | None = None,
    ) -> GenerationResult:
        return self.complete(self.build_chat_prompt(messages), generation=generation)

    # ------------------------------------------------------------------ scoring

    @torch.inference_mode()
    def logprob(self, text: str) -> dict[str, float]:
        """Mean token log-probability of `text` under the model.

        Used by evaluation and by the API's scoring path. Reported as a
        measurement with its token count, so it can be compared across runs.
        """
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < 2:
            raise EngineError("need at least 2 tokens to score")
        ids = ids[: self.max_context]

        tensor = torch.tensor([ids], dtype=torch.long, device=self.device)
        output = self.model(input_ids=tensor, labels=tensor)
        loss = float(output.loss.item())
        return {
            "tokens": len(ids),
            "mean_nll": loss,
            "mean_logprob": -loss,
            "perplexity": float(torch.exp(torch.tensor(loss)).item()),
        }
