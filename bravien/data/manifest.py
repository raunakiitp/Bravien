"""Dataset manifests: provenance, statistics, and reproducibility (§12, §14, §23).

Every pipeline stage emits a manifest. A manifest is the record that makes a
training run reproducible and a dataset claim auditable — if the pipeline did not
measure it, it does not go in here.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class SourceRecord:
    """Provenance for one dataset source (§14).

    `license` is mandatory and free-text on purpose: an unknown license should
    be recorded as "unknown" and reviewed, never quietly omitted.
    """

    name: str
    url: str
    license: str
    retrieved_at: str = field(default_factory=utc_now)
    processing: list[str] = field(default_factory=list)
    documents: int = 0
    bytes: int = 0
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.license:
            raise ValueError(
                f"source {self.name!r} has no license recorded; use 'unknown' "
                f"explicitly if it must be reviewed"
            )


@dataclass
class StageStats:
    """What one pipeline stage did to the corpus."""

    stage: str
    documents_in: int = 0
    documents_out: int = 0
    characters_in: int = 0
    characters_out: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    notes: str = ""

    @property
    def documents_dropped(self) -> int:
        return self.documents_in - self.documents_out

    @property
    def drop_rate(self) -> float:
        if self.documents_in == 0:
            return 0.0
        return self.documents_dropped / self.documents_in


@dataclass
class Manifest:
    """A complete, checksummed description of a prepared dataset."""

    name: str
    stage: str
    created_at: str = field(default_factory=utc_now)
    documents: int = 0
    characters: int = 0
    tokens: int = 0
    sequences: int = 0
    sequence_length: int = 0
    languages: dict[str, int] = field(default_factory=dict)
    stages: list[StageStats] = field(default_factory=list)
    sources: list[SourceRecord] = field(default_factory=list)
    files: list[dict[str, Any]] = field(default_factory=list)
    tokenizer: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    content_checksum: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def record_environment(self) -> None:
        """Capture the interpreter and platform for reproducibility (§23)."""
        self.environment = {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
        }
        try:
            import torch  # imported lazily: the data stages do not require it

            self.environment["torch"] = torch.__version__
            self.environment["cuda"] = getattr(torch.version, "cuda", None)
        except ImportError:
            self.environment["torch"] = None

    def add_stage(self, stats: StageStats) -> None:
        self.stages.append(stats)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> Manifest:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        stages = [StageStats(**s) for s in data.pop("stages", [])]
        sources = [SourceRecord(**s) for s in data.pop("sources", [])]
        manifest = cls(**data)
        manifest.stages = stages
        manifest.sources = sources
        return manifest

    def checksum(self) -> str:
        """Hash of the manifest itself, excluding its own checksum field.

        Training checkpoints record this so a run can always be traced back to
        the exact dataset that produced it.
        """
        payload = self.to_dict()
        payload.pop("content_checksum", None)
        payload.pop("created_at", None)
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
        return "sha256:" + hashlib.sha256(blob).hexdigest()[:32]

    def summary(self) -> str:
        lines = [
            f"Dataset manifest: {self.name}  (stage: {self.stage})",
            f"  created:      {self.created_at}",
            f"  documents:    {self.documents:,}",
            f"  characters:   {self.characters:,}",
        ]
        if self.tokens:
            lines.append(f"  tokens:       {self.tokens:,}")
        if self.sequences:
            lines.append(
                f"  sequences:    {self.sequences:,} x {self.sequence_length} tokens"
            )
        if self.languages:
            langs = ", ".join(
                f"{k}={v:,}" for k, v in sorted(
                    self.languages.items(), key=lambda kv: -kv[1]
                )[:6]
            )
            lines.append(f"  languages:    {langs}")
        if self.stages:
            lines.append("")
            lines.append("  Pipeline")
            for st in self.stages:
                detail = ""
                if st.dropped:
                    detail = "  [" + ", ".join(
                        f"{k}:{v:,}" for k, v in sorted(
                            st.dropped.items(), key=lambda kv: -kv[1]
                        )
                    ) + "]"
                lines.append(
                    f"    {st.stage:<16} {st.documents_in:>8,} -> "
                    f"{st.documents_out:>8,}  ({st.drop_rate * 100:5.1f}% dropped)"
                    f"{detail}"
                )
        if self.sources:
            lines.append("")
            lines.append("  Sources")
            for src in self.sources:
                lines.append(
                    f"    {src.name}  [{src.license}]  {src.documents:,} docs"
                )
        return "\n".join(lines)


def file_checksum(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Streaming sha256 of a file, so large shards do not need to be buffered."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()
