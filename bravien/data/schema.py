"""Canonical Bravien training data schema and serialization.

Defines the universal format for all training data in the Bravien system:
- Model-independent instruction and multi-turn conversational format
- Compatible with Qwen2.5 and future Bravien chat templates
- Strict structural validation and content fingerprinting
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]
VALID_ROLES = {"system", "user", "assistant", "tool"}

VALID_CATEGORIES = {
    "general_knowledge",
    "instruction_following",
    "dialogue",
    "reasoning",
    "coding",
    "safety_refusal",
    "bravien_assistant",
    "tool_use",
    "hinglish",
}


class ValidationError(ValueError):
    """Raised when a training example violates schema or structural rules."""
    pass


@dataclass
class BravienMessage:
    role: Role
    content: str
    name: str | None = None

    def __post_init__(self) -> None:
        if self.role not in VALID_ROLES:
            raise ValidationError(f"Invalid message role: {self.role!r}. Allowed: {VALID_ROLES}")
        if not isinstance(self.content, str):
            raise ValidationError(f"Message content must be a string, got {type(self.content).__name__}")
        if not self.content.strip():
            raise ValidationError("Message content must not be empty or whitespace only")

    def to_dict(self) -> dict[str, str]:
        res = {"role": self.role, "content": self.content}
        if self.name:
            res["name"] = self.name
        return res

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BravienMessage:
        role = data.get("role")
        content = data.get("content")
        if role is None or content is None:
            raise ValidationError(f"Missing required keys in message dict: {data}")
        return cls(role=role, content=content, name=data.get("name"))


@dataclass
class BravienTrainingExample:
    id: str
    source: str
    category: str
    messages: list[BravienMessage]
    quality_score: float = 1.0
    language: str = "en"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = str(uuid.uuid4())
        self.validate()

    def validate(self) -> None:
        """Validate structural invariants."""
        if not self.messages:
            raise ValidationError("Training example must have at least one message")

        # Must have at least one user turn
        user_turns = [m for m in self.messages if m.role == "user"]
        if not user_turns:
            raise ValidationError("Training example must contain at least one user message")

        # Must end with an assistant turn (the target supervision)
        if self.messages[-1].role != "assistant":
            raise ValidationError(
                f"Last message in training example must be assistant, got {self.messages[-1].role!r}"
            )

        # Quality score range
        if not (0.0 <= self.quality_score <= 1.0):
            raise ValidationError(f"quality_score must be between 0.0 and 1.0, got {self.quality_score}")

        # Ensure metadata is dict
        if not isinstance(self.metadata, dict):
            self.metadata = {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "category": self.category,
            "messages": [m.to_dict() for m in self.messages],
            "quality_score": round(self.quality_score, 4),
            "language": self.language,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BravienTrainingExample:
        messages_raw = data.get("messages", [])
        if not isinstance(messages_raw, list):
            raise ValidationError(f"'messages' must be a list, got {type(messages_raw).__name__}")

        messages = [
            BravienMessage.from_dict(m) if isinstance(m, dict) else m
            for m in messages_raw
        ]

        return cls(
            id=data.get("id") or str(uuid.uuid4()),
            source=str(data.get("source", "unknown")),
            category=str(data.get("category", "instruction_following")),
            messages=messages,
            quality_score=float(data.get("quality_score", 1.0)),
            language=str(data.get("language", "en")),
            metadata=data.get("metadata", {}),
        )

    def content_fingerprint(self) -> str:
        """Normalized cryptographic hash of the conversation content for exact deduplication."""
        normalized_parts = []
        for m in self.messages:
            # Normalize whitespace and case for robust fingerprinting
            norm_content = re.sub(r"\s+", " ", m.content.strip().lower())
            normalized_parts.append(f"{m.role}:{norm_content}")
        raw = "\n".join(normalized_parts)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def user_prompt_fingerprint(self) -> str:
        """Hash of only the user inputs."""
        user_parts = [
            re.sub(r"\s+", " ", m.content.strip().lower())
            for m in self.messages
            if m.role == "user"
        ]
        return hashlib.sha256("\n".join(user_parts).encode("utf-8")).hexdigest()

    def estimate_tokens(self) -> int:
        """Heuristic character-to-token estimate (~4 chars per token)."""
        total_chars = sum(len(m.content) for m in self.messages)
        return max(1, total_chars // 4)
