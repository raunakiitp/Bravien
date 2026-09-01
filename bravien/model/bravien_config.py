"""Native Bravien model configuration.

Defines the architecture specification for native Bravien decoder-only
causal transformers, including standard and ~1.5B target configurations.
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
    """Architecture configuration for a native Bravien Transformer model."""

    # Vocabulary & Tokens
    vocab_size: int = 32000
    tie_word_embeddings: bool = True
    pad_token_id: int = 0
    unk_token_id: int = 1
    bos_token_id: int = 2
    eos_token_id: int = 3

    # Core Dimensions (Default: Bravien-1.5B reference configuration)
    hidden_size: int = 1536
    num_layers: int = 28
    num_heads: int = 16
    num_kv_heads: int = 4  # 4:1 GQA ratio for high inference throughput & reduced KV cache
    intermediate_size: int = 4096  # SwiGLU expansion (~2.67x hidden_size)

    # Context Length
    max_position_embeddings: int = 4096

    # Normalization & Position Encoding
    norm_kind: NormKind = "rmsnorm"
    norm_eps: float = 1e-5
    position_encoding: PosEncoding = "rope"
    rope_theta: float = 10000.0
    activation: Activation = "silu"

    # Regularization & Initialization
    dropout: float = 0.0
    attention_dropout: float = 0.0
    initializer_range: float = 0.02

    # Precision & Execution
    dtype: str = "bfloat16"
    use_gradient_checkpointing: bool = False

    # Metadata & Versioning
    model_type: str = "bravien"
    architecture: str = "BravienForCausalLM"
    architectures: list[str] = field(default_factory=lambda: ["BravienForCausalLM"])
    architecture_version: str = "1.0.0"
    name: str = "bravien-1.5b"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def kv_dim(self) -> int:
        return self.head_dim * self.num_kv_heads

    @property
    def num_kv_groups(self) -> int:
        return self.num_heads // self.num_kv_heads

    def validate(self) -> None:
        """Validate consistency of hyperparameters before allocating any tensors."""
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
                f"num_kv_heads ({self.num_kv_heads}) cannot exceed num_heads ({self.num_heads})"
            )
        if self.head_dim % 2 != 0 and self.position_encoding == "rope":
            raise ValueError(
                f"head_dim ({self.head_dim}) must be even for rotary position embeddings"
            )
        for dim_name in ("vocab_size", "hidden_size", "num_layers", "num_heads",
                         "intermediate_size", "max_position_embeddings"):
            if getattr(self, dim_name) <= 0:
                raise ValueError(f"{dim_name} must be positive, got {getattr(self, dim_name)}")

    @property
    def version(self) -> str:
        return self.architecture_version

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def replace(self, **kwargs: Any) -> BravienConfig:
        """Return a new BravienConfig with updated field values."""
        d = self.to_dict()
        d.update(kwargs)
        return BravienConfig.from_dict(d)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BravienConfig:
        unknown = [k for k in data.keys() if k not in cls.__dataclass_fields__]
        if unknown:
            raise ValueError(f"Unknown configuration key(s): {', '.join(unknown)}")
        return cls(**data)

    def save(self, path: str | Path) -> Path:
        """Save configuration directly to a JSON file path."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return p

    def save_json(self, path: str | Path) -> Path:
        return self.save(path)

    @classmethod
    def load(cls, path: str | Path) -> BravienConfig:
        """Load configuration directly from a JSON file path."""
        p = Path(path)
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_json_file(cls, path: str | Path) -> BravienConfig:
        return cls.load(path)

    def save_pretrained(self, save_directory: str | Path) -> Path:
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        config_file = save_path / "model_config.json"
        return self.save(config_file)

    @classmethod
    def from_pretrained(cls, save_directory: str | Path) -> BravienConfig:
        load_path = Path(save_directory)
        config_file = load_path / "model_config.json"
        if not config_file.exists():
            config_file = load_path / "config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"No configuration file found at {load_path}")
        return cls.load(config_file)



# Standard Architectural Presets
BRAVIEN_PRESETS: dict[str, BravienConfig] = {
    # Tiny model for unit testing, fast CPU smoke tests, and CI/CD validation (~0.6M params)
    "bravien-tiny": BravienConfig(
        name="bravien-tiny",
        vocab_size=4000,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=256,
        max_position_embeddings=2048,
    ),
    # Small model for local laptop testing (Stage 1-5 style, ~40M params)
    "bravien-small": BravienConfig(
        name="bravien-small",
        vocab_size=32000,
        hidden_size=512,
        num_layers=8,
        num_heads=8,
        num_kv_heads=4,
        intermediate_size=1376,
        max_position_embeddings=1024,
    ),
    # ~1.0B Candidate (24 layers, hidden 2048, 16/4 heads, SwiGLU 5632) -> ~1.14B params
    "bravien-1.0b": BravienConfig(
        name="bravien-1.0b",
        vocab_size=32000,
        hidden_size=2048,
        num_layers=22,
        num_heads=16,
        num_kv_heads=4,
        intermediate_size=5632,
        max_position_embeddings=2048,
        tie_word_embeddings=True,
    ),
    # ~1.2B Candidate (26 layers, hidden 2048, 16/4 heads, SwiGLU 5632) -> ~1.24B params
    "bravien-1.2b": BravienConfig(
        name="bravien-1.2b",
        vocab_size=32000,
        hidden_size=2048,
        num_layers=26,
        num_heads=16,
        num_kv_heads=4,
        intermediate_size=5632,
        max_position_embeddings=4096,
        tie_word_embeddings=True,
    ),
    # ~1.5B Target Architecture (32 layers, hidden 2048, 16/4 heads, SwiGLU 5632) -> ~1.51B params
    "bravien-1.5b": BravienConfig(
        name="bravien-1.5b",
        vocab_size=32000,
        hidden_size=2048,
        num_layers=32,
        num_heads=16,
        num_kv_heads=4,
        intermediate_size=5632,
        max_position_embeddings=4096,
        tie_word_embeddings=True,
    ),
    # ~1.7B Candidate (32 layers, hidden 2048, 16/4 heads, SwiGLU 5632, Untied Embeddings + 64k Vocab) -> ~1.71B params
    "bravien-1.7b": BravienConfig(
        name="bravien-1.7b",
        vocab_size=64000,
        hidden_size=2048,
        num_layers=32,
        num_heads=16,
        num_kv_heads=4,
        intermediate_size=5632,
        max_position_embeddings=4096,
        tie_word_embeddings=False,
    ),
}


def get_bravien_preset(name: str) -> BravienConfig:
    """Retrieve a pre-configured architecture by name."""
    normalized = name.lower().strip()
    if normalized in BRAVIEN_PRESETS:
        return BRAVIEN_PRESETS[normalized]
    
    # Try with "bravien-" prefix
    if not normalized.startswith("bravien-"):
        prefixed = f"bravien-{normalized}"
        if prefixed in BRAVIEN_PRESETS:
            return BRAVIEN_PRESETS[prefixed]

    # Try without "bravien-" prefix
    if normalized.startswith("bravien-"):
        unprefixed = normalized.removeprefix("bravien-")
        if unprefixed in BRAVIEN_PRESETS:
            return BRAVIEN_PRESETS[unprefixed]

    available = ", ".join(BRAVIEN_PRESETS.keys())
    raise KeyError(f"Unknown preset '{name}'. Available presets: {available}")

