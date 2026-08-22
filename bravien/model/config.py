"""Bravien model configuration.

A single dataclass describes the whole architecture. Everything the brief
requires to be configurable (§7) lives here, and nothing about the shape of the
network is hardcoded elsewhere.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

NormKind = Literal["rmsnorm", "layernorm"]
PosEncoding = Literal["rope", "learned", "none"]
Activation = Literal["silu", "gelu"]


@dataclass
class BravienConfig:
    """Architecture of a Bravien decoder-only transformer.

    Defaults describe *Bravien Small*, the reference configuration sized to
    train on a single 6 GB consumer GPU.
    """

    # Vocabulary / embeddings
    vocab_size: int = 32000
    tie_word_embeddings: bool = True

    # Reserved token ids. Mirrored from the tokenizer at build time so a
    # checkpoint's config alone is enough to know where generation must stop —
    # the inference engine should not have to guess if a tokenizer is absent.
    pad_token_id: int = 0
    unk_token_id: int = 1
    bos_token_id: int = 2
    eos_token_id: int = 3

    # Core dimensions
    hidden_size: int = 512
    num_layers: int = 8
    num_heads: int = 8
    num_kv_heads: int = 4
    intermediate_size: int = 1376

    # Sequence
    max_position_embeddings: int = 1024

    # Regularisation
    dropout: float = 0.0
    attention_dropout: float = 0.0

    # Component choices
    norm_kind: NormKind = "rmsnorm"
    norm_eps: float = 1e-5
    position_encoding: PosEncoding = "rope"
    rope_theta: float = 10000.0
    activation: Activation = "silu"

    # Initialisation
    initializer_range: float = 0.02

    # Identity — surfaced by the runtime and the UI, never invented there.
    name: str = "bravien-small"
    version: str = "0.1.0"

    # Free-form provenance (tokenizer hash, dataset manifest, ...)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    # ---------------------------------------------------------------- checks

    def validate(self) -> None:
        """Fail loudly on an inconsistent architecture, before any allocation."""
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads}) for grouped-query attention"
            )
        if self.num_kv_heads > self.num_heads:
            raise ValueError(
                f"num_kv_heads ({self.num_kv_heads}) cannot exceed "
                f"num_heads ({self.num_heads})"
            )
        if self.head_dim % 2 != 0 and self.position_encoding == "rope":
            raise ValueError(
                f"head_dim ({self.head_dim}) must be even to apply RoPE"
            )
        for name in ("vocab_size", "hidden_size", "num_layers", "num_heads",
                     "intermediate_size", "max_position_embeddings"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if not 0.0 <= self.attention_dropout < 1.0:
            raise ValueError(
                f"attention_dropout must be in [0, 1), got {self.attention_dropout}"
            )
        for name in ("pad_token_id", "unk_token_id", "bos_token_id", "eos_token_id"):
            value = getattr(self, name)
            if not 0 <= value < self.vocab_size:
                raise ValueError(
                    f"{name}={value} is outside the vocabulary of "
                    f"{self.vocab_size}"
                )

    # ------------------------------------------------------------ derived

    def replace(self, **changes: Any) -> BravienConfig:
        """A copy with fields overridden, re-validated.

        Used when the tokenizer's real vocabulary supersedes a config file's
        guess: mutating in place would leave an already-built model disagreeing
        with its own config.
        """
        unknown = set(changes) - set(self.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return BravienConfig.from_dict({**self.to_dict(), **changes})

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def num_kv_groups(self) -> int:
        """How many query heads share each key/value head."""
        return self.num_heads // self.num_kv_heads

    # ------------------------------------------------------ serialisation

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BravienConfig:
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**data)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> BravienConfig:
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix in (".yaml", ".yml"):
            import yaml  # local import: YAML is optional

            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        # Allow a nested {"model": {...}} document.
        if isinstance(data, dict) and "model" in data and isinstance(data["model"], dict):
            data = data["model"]
        return cls.from_dict(data)


# --------------------------------------------------------------- presets

def _preset_tiny() -> BravienConfig:
    """Deliberately minute: for unit tests, CI, and the CPU smoke test."""
    return BravienConfig(
        name="bravien-tiny",
        vocab_size=8192,
        hidden_size=256,
        num_layers=4,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=688,
        max_position_embeddings=512,
    )


def _preset_small() -> BravienConfig:
    """Reference config — trains on a single 6 GB GPU."""
    return BravienConfig(name="bravien-small")


def _preset_base() -> BravienConfig:
    """Next step up; needs either a larger card or gradient accumulation."""
    return BravienConfig(
        name="bravien-base",
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        num_kv_heads=4,
        intermediate_size=2048,
        max_position_embeddings=2048,
    )


PRESETS: dict[str, Any] = {
    "tiny": _preset_tiny,
    "small": _preset_small,
    "base": _preset_base,
}


def get_preset(name: str) -> BravienConfig:
    key = name.lower().removeprefix("bravien-")
    if key not in PRESETS:
        raise KeyError(
            f"Unknown preset {name!r}. Available: {sorted(PRESETS)}"
        )
    return PRESETS[key]()
