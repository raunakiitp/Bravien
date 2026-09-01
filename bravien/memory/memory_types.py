"""Memory record definitions and categories."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class MemoryType(str, Enum):
    CONVERSATION = "conversation"
    PREFERENCE = "preference"
    PROJECT = "project"
    FACT = "fact"
    SESSION = "session"


@dataclass
class MemoryRecord:
    id: str
    key: str
    value: str
    type: MemoryType
    confidence: float = 1.0
    importance: float = 1.0
    source: str = "user_explicit"
    tags: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "key": self.key,
            "value": self.value,
            "type": self.type.value,
            "confidence": self.confidence,
            "importance": self.importance,
            "source": self.source,
            "tags": self.tags,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
