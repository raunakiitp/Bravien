"""Intelligent tool selection determining optimal execution path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bravien.agent.intent import IntentCategory, classify_intent
from bravien.tools.registry import ToolRegistry, tool_registry


@dataclass
class ToolSelectionDecision:
    action: str  # "ANSWER_DIRECT", "USE_TOOL", "MULTI_STEP", "ASK_CLARIFICATION", "REFUSE"
    selected_tool: str | None
    arguments: dict[str, Any]
    requires_confirmation: bool
    reason: str


class ToolSelector:
    """Selects tools based on hybrid deterministic heuristics and safety policies."""

    def __init__(self, registry: ToolRegistry | None = None) -> None:
        self.registry = registry or tool_registry

    def select(self, query: str, context: dict[str, Any] | None = None) -> ToolSelectionDecision:
        intent_res = classify_intent(query, context)

        # 1. Unsafe Refusal
        if intent_res.intent == IntentCategory.UNSAFE_REQUEST:
            return ToolSelectionDecision(
                action="REFUSE",
                selected_tool=None,
                arguments={},
                requires_confirmation=False,
                reason="Unsafe request rejected by safety boundary policy.",
            )

        # 2. Direct Arithmetic
        if intent_res.intent == IntentCategory.MATHEMATICS:
            # Extract expression
            expr = query.strip()
            # Clean common prefixes
            for prefix in ("what is", "calculate", "solve", "evaluate"):
                if expr.lower().startswith(prefix):
                    expr = expr[len(prefix):].strip().rstrip("?")
            return ToolSelectionDecision(
                action="USE_TOOL",
                selected_tool="calculator",
                arguments={"expression": expr},
                requires_confirmation=False,
                reason="Direct arithmetic calculation detected.",
            )

        # 3. Memory Read / Write
        if intent_res.intent == IntentCategory.MEMORY:
            if "remember" in query.lower() or "save" in query.lower() or "prefer" in query.lower():
                return ToolSelectionDecision(
                    action="USE_TOOL",
                    selected_tool="memory_write",
                    arguments={"key": "preference", "value": query.strip(), "category": "preference"},
                    requires_confirmation=False,
                    reason="User preference store instruction.",
                )
            return ToolSelectionDecision(
                action="USE_TOOL",
                selected_tool="memory_read",
                arguments={"query": query.strip(), "category": "preference"},
                requires_confirmation=False,
                reason="Memory retrieval instruction.",
            )

        # 4. Document / RAG Query
        if intent_res.intent == IntentCategory.DOCUMENT_QUESTION:
            return ToolSelectionDecision(
                action="USE_TOOL",
                selected_tool="document_retrieval",
                arguments={"query": query.strip(), "top_k": 3},
                requires_confirmation=False,
                reason="Document grounding retrieval instruction.",
            )

        # 5. Planning
        if intent_res.intent == IntentCategory.PLANNING:
            return ToolSelectionDecision(
                action="USE_TOOL",
                selected_tool="structured_planning",
                arguments={"goal": query.strip(), "max_steps": 4},
                requires_confirmation=False,
                reason="Planning goal decomposition instruction.",
            )

        # 6. Multi-Step Task
        if intent_res.intent == IntentCategory.MULTI_STEP_TASK:
            return ToolSelectionDecision(
                action="MULTI_STEP",
                selected_tool=None,
                arguments={"task": query.strip()},
                requires_confirmation=False,
                reason="Sequenced multi-step task execution loop.",
            )

        # 7. Default Direct Answer
        return ToolSelectionDecision(
            action="ANSWER_DIRECT",
            selected_tool=None,
            arguments={},
            requires_confirmation=False,
            reason="Conversational or knowledge query suitable for direct response.",
        )


# Default global tool selector instance
tool_selector = ToolSelector()
