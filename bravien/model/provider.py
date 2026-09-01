"""Abstract Model Provider Interface decoupling agent logic from specific model backends."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from bravien.model.native_provider import (
    BravienNativeProvider,
    NativeGenerationConfig,
)


@dataclass
class GenerationConfig:
    temperature: float = 0.4
    top_p: float = 0.9
    top_k: int | None = 40
    max_new_tokens: int = 512
    repetition_penalty: float = 1.05
    stop_tokens: list[str] = field(default_factory=list)
    seed: int | None = None


@dataclass
class ModelMetadata:
    name: str
    checkpoint_path: str
    device: str
    precision: str
    context_window: int
    parameter_count: int
    provider_type: str = "native"
    local_only: bool = True
    architecture: str = "bravien"


class ModelProvider(ABC):
    """Abstract interface for model providers (native, local fallback, mock)."""

    @abstractmethod
    def load(self, checkpoint_path: str, device: str = "cuda", precision: str = "auto") -> None:
        """Loads model weights and tokenizer into memory."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unloads model weights from memory."""
        pass

    @abstractmethod
    def generate(self, prompt: str | list[dict[str, str]], config: GenerationConfig | None = None) -> str:
        """Generates a complete response text."""
        pass

    @abstractmethod
    def stream(self, prompt: str | list[dict[str, str]], config: GenerationConfig | None = None) -> Iterator[str]:
        """Streams generated tokens."""
        pass

    @abstractmethod
    def health(self) -> dict[str, Any]:
        """Returns health and status dictionary."""
        pass

    @abstractmethod
    def metadata(self) -> ModelMetadata:
        """Returns metadata about the active model."""
        pass


class BravienLocalProvider(ModelProvider):
    """Optional historical/rollback Qwen-compatible provider."""

    def __init__(self, checkpoint_path: str = "checkpoints/bravien-v3") -> None:
        self.checkpoint_path = checkpoint_path
        self._engine: Any = None

    def _ensure_engine(self) -> Any:
        if self._engine is None:
            from bravien.inference.hf_engine import HFInferenceEngine
            self._engine = HFInferenceEngine.from_pretrained(self.checkpoint_path)
        return self._engine

    def load(self, checkpoint_path: str, device: str = "cuda", precision: str = "auto") -> None:
        from bravien.inference.hf_engine import HFInferenceEngine
        self.checkpoint_path = checkpoint_path
        self._engine = HFInferenceEngine.from_pretrained(checkpoint_path, device=device)

    def unload(self) -> None:
        if self._engine:
            self._engine = None

    def generate(self, prompt: str | list[dict[str, str]], config: GenerationConfig | None = None) -> str:
        engine = self._ensure_engine()
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        res = engine.chat(messages=messages, generation={"max_new_tokens": config.max_new_tokens if config else 512})
        return res.text

    def stream(self, prompt: str | list[dict[str, str]], config: GenerationConfig | None = None) -> Iterator[str]:
        engine = self._ensure_engine()
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        for chunk in engine.chat_stream(messages=messages, generation={"max_new_tokens": config.max_new_tokens if config else 512}):
            yield chunk

    def health(self) -> dict[str, Any]:
        engine = self._ensure_engine()
        return {
            "status": "ok",
            "backend": "qwen_compat",
            "model_loaded": engine.model is not None,
            "device": str(engine.device),
            "checkpoint": self.checkpoint_path,
        }

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            name="Bravien Local (v3 Rollback Baseline)",
            checkpoint_path=self.checkpoint_path,
            device="cuda",
            precision="bfloat16",
            context_window=2048,
            parameter_count=494032768,
            provider_type="qwen_compat",
            local_only=True,
            architecture="qwen2",
        )


def get_model_provider(
    backend: str | None = None,
    checkpoint_path: str | None = None,
) -> ModelProvider | BravienNativeProvider:
    """Factory creating the appropriate ModelProvider.

    DEFAULT: Native Bravien 1.5B (native backend).
    Qwen compatibility is DISABLED by default and only available if
    BRAVIEN_ENABLE_QWEN_COMPAT=true or explicit rollback is requested.
    """
    chosen_backend = (
        backend
        or os.environ.get("BRAVIEN_MODEL_BACKEND", "native")
    ).lower()

    if chosen_backend == "native":
        ckpt = checkpoint_path or os.environ.get("BRAVIEN_NATIVE_CHECKPOINT", "checkpoints/bravien-native-1.5b")
        return BravienNativeProvider(checkpoint_path=ckpt)
    elif chosen_backend in ("qwen_compat", "hf", "legacy"):
        enable_compat = os.environ.get("BRAVIEN_ENABLE_QWEN_COMPAT", "false").lower() in ("true", "1")
        if not enable_compat and not backend:
            raise RuntimeError(
                "Qwen compatibility backend is disabled in default production configuration. "
                "Set BRAVIEN_ENABLE_QWEN_COMPAT=true to enable historical rollback mode."
            )
        ckpt = checkpoint_path or os.environ.get("BRAVIEN_CHECKPOINT_PATH", "checkpoints/bravien-v3")
        return BravienLocalProvider(checkpoint_path=ckpt)
    else:
        raise ValueError(f"Unknown model provider backend '{chosen_backend}'. Supported: 'native', 'qwen_compat'")
