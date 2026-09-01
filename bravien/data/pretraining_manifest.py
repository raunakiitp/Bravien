"""Pretraining data manifest management for native Bravien pretraining corpora."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

DomainType = Literal[
    "general_web", "books", "code", "math_stem", "multilingual_hindi", "hinglish",
    "technical_docs", "instruction_sft", "safety", "agent_tools"
]


@dataclass
class CorpusSource:
    """Metadata for a single data source shard."""

    source_id: str
    name: str
    domain: DomainType
    language: str  # e.g., "en", "hi", "hinglish", "code"
    estimated_tokens: int
    sample_count: int
    license: str
    file_path: str
    sha256_checksum: str
    quality_score: float = 1.0  # 0.0 to 1.0
    is_synthetic: bool = False
    contamination_flag: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CorpusSource:
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class PretrainingManifest:
    """Master manifest tracking all vetted data sources for native Bravien pretraining."""

    manifest_version: str = "1.0.0"
    created_at: str = ""
    total_estimated_tokens: int = 0
    total_samples: int = 0
    sources: list[CorpusSource] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_source(self, source: CorpusSource) -> None:
        self.sources.append(source)
        self.total_estimated_tokens += source.estimated_tokens
        self.total_samples += source.sample_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            "created_at": self.created_at,
            "total_estimated_tokens": self.total_estimated_tokens,
            "total_samples": self.total_samples,
            "sources": [s.to_dict() for s in self.sources],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PretrainingManifest:
        sources = [CorpusSource.from_dict(s) for s in data.get("sources", [])]
        return cls(
            manifest_version=data.get("manifest_version", "1.0.0"),
            created_at=data.get("created_at", ""),
            total_estimated_tokens=data.get("total_estimated_tokens", 0),
            total_samples=data.get("total_samples", 0),
            sources=sources,
            metadata=data.get("metadata", {}),
        )

    def save(self, path: str | Path) -> None:
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> PretrainingManifest:
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
