"""Relevance scoring and selective memory retrieval."""

from __future__ import annotations

import re
from typing import Any

from bravien.memory.memory_store import MemoryStore, memory_store
from bravien.memory.memory_types import MemoryRecord, MemoryType


class MemoryRetriever:
    """Retrieves and ranks relevant memory records without injecting entire database into context."""

    def __init__(self, store: MemoryStore | None = None) -> None:
        self.store = store or memory_store

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        memory_type: MemoryType | None = None,
        min_relevance: float = 0.2,
    ) -> list[tuple[MemoryRecord, float]]:
        records = self.store.list_all(memory_type)
        if not records:
            return []

        query_tokens = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", query.lower()))
        scored: list[tuple[MemoryRecord, float]] = []

        for r in records:
            rec_text = f"{r.key} {r.value} {' '.join(r.tags)}".lower()
            rec_tokens = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", rec_text))

            if not query_tokens:
                score = r.importance * 0.5
            else:
                overlap = len(query_tokens.intersection(rec_tokens))
                if overlap == 0:
                    score = 0.0
                else:
                    score = (overlap / len(query_tokens)) * 0.7 + (r.importance * 0.3)

            if score >= min_relevance:
                scored.append((r, round(score, 3)))

        # Sort descending by score
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def format_for_context(self, query: str, top_k: int = 3) -> str:
        results = self.retrieve(query, top_k=top_k)
        if not results:
            return ""

        lines = ["--- RELEVANT USER CONTEXT & PREFERENCES ---"]
        for record, score in results:
            lines.append(f"- [{record.type.value.upper()}] {record.key}: {record.value}")
        return "\n".join(lines)


# Default global memory retriever
memory_retriever = MemoryRetriever()
