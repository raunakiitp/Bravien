"""Structured in-memory and persistent memory store."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from bravien.memory.memory_policy import MemoryPolicy
from bravien.memory.memory_types import MemoryRecord, MemoryType


class MemoryStore:
    """Stores and manages structured memory records with safety validation."""

    def __init__(self) -> None:
        self._records: dict[str, MemoryRecord] = {}
        self._counter: int = 0

    def save(
        self,
        key: str,
        value: str,
        memory_type: MemoryType = MemoryType.PREFERENCE,
        confidence: float = 1.0,
        importance: float = 1.0,
        tags: list[str] | None = None,
    ) -> tuple[bool, str, MemoryRecord | None]:
        is_safe, reason = MemoryPolicy.is_safe_to_store(key, value)
        if not is_safe:
            return False, reason, None

        # Check for existing record with the same key to update
        for existing in self._records.values():
            if existing.key.lower() == key.lower() and existing.type == memory_type:
                existing.value = value
                existing.confidence = confidence
                existing.importance = importance
                existing.tags = tags or existing.tags
                existing.updated_at = datetime.now(timezone.utc).isoformat()
                return True, "Memory updated successfully.", existing

        self._counter += 1
        rec_id = f"mem-{int(time.time() * 1000)}-{self._counter}"
        record = MemoryRecord(
            id=rec_id,
            key=key,
            value=value,
            type=memory_type,
            confidence=confidence,
            importance=importance,
            tags=tags or [],
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        self._records[rec_id] = record
        return True, "Memory stored successfully.", record

    def get_by_id(self, record_id: str) -> MemoryRecord | None:
        return self._records.get(record_id)

    def list_all(self, memory_type: MemoryType | None = None) -> list[MemoryRecord]:
        if memory_type is None:
            return list(self._records.values())
        return [r for r in self._records.values() if r.type == memory_type]

    def delete(self, record_id: str) -> bool:
        if record_id in self._records:
            del self._records[record_id]
            return True
        return False

    def clear(self) -> None:
        self._records.clear()


# Default global memory store
memory_store = MemoryStore()
