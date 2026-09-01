"""Tool definition schemas and metadata contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class ToolParameter:
    name: str
    type: str  # "string", "number", "boolean", "object", "array"
    description: str
    required: bool = True
    default: Any = None
    enum: list[Any] | None = None


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: list[ToolParameter]
    handler: Callable[..., Any]
    risk_level: RiskLevel = RiskLevel.LOW
    deterministic: bool = True
    requires_confirmation: bool = False
    timeout_seconds: float = 5.0
    tags: list[str] = field(default_factory=list)

    def to_schema(self) -> dict[str, Any]:
        properties: dict[str, Any] = {}
        required: list[str] = []

        for p in self.parameters:
            prop: dict[str, Any] = {
                "type": p.type,
                "description": p.description,
            }
            if p.enum:
                prop["enum"] = p.enum
            if p.default is not None:
                prop["default"] = p.default
            properties[p.name] = prop
            if p.required:
                required.append(p.name)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
            "risk_level": self.risk_level.value,
            "deterministic": self.deterministic,
            "requires_confirmation": self.requires_confirmation,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass
class ToolExecutionResult:
    tool: str
    arguments: dict[str, Any]
    status: str  # "SUCCESS", "FAILED", "BLOCKED", "TIMEOUT"
    result: Any
    duration_ms: float
    verification_status: str  # "VERIFIED", "UNVERIFIED", "FAILED"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "arguments": self.arguments,
            "status": self.status,
            "result": self.result,
            "duration_ms": round(self.duration_ms, 2),
            "verification_status": self.verification_status,
            "error": self.error,
        }
