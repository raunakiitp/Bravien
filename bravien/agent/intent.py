"""Deterministic and heuristic intent classification layer for Bravien."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class IntentCategory(str, Enum):
    CONVERSATION = "conversation"
    FACTUAL_QUESTION = "factual_question"
    REASONING = "reasoning"
    MATHEMATICS = "mathematics"
    CODING = "coding"
    DOCUMENT_QUESTION = "document_question"
    MEMORY = "memory"
    PLANNING = "planning"
    TOOL_REQUIRED = "tool_required"
    MULTI_STEP_TASK = "multi_step_task"
    CLARIFICATION_REQUIRED = "clarification_required"
    UNSAFE_REQUEST = "unsafe_request"
    UNKNOWN = "unknown"


@dataclass
class IntentResult:
    intent: IntentCategory
    confidence: float
    requires_tool: bool
    requires_memory: bool
    requires_rag: bool
    requires_reasoning: bool
    requires_confirmation: bool
    safety_level: str  # "SAFE", "CAUTION", "UNSAFE"
    target_tool: str | None = None
    extracted_entities: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent.value,
            "confidence": self.confidence,
            "requires_tool": self.requires_tool,
            "requires_memory": self.requires_memory,
            "requires_rag": self.requires_rag,
            "requires_reasoning": self.requires_reasoning,
            "requires_confirmation": self.requires_confirmation,
            "safety_level": self.safety_level,
            "target_tool": self.target_tool,
            "extracted_entities": self.extracted_entities,
            "reasoning": self.reasoning,
        }


@dataclass
class TaskSpec:
    intent: IntentCategory
    complexity: str  # "LOW", "MEDIUM", "HIGH"
    requires_model: bool
    requires_tool: bool
    requires_memory: bool
    requires_rag: bool
    requires_verification: bool
    risk_level: str  # "LOW", "MEDIUM", "HIGH"
    expected_output: str
    max_steps: int = 8
    timeout_seconds: float = 30.0
    confirmation_required: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent.value,
            "complexity": self.complexity,
            "requires_model": self.requires_model,
            "requires_tool": self.requires_tool,
            "requires_memory": self.requires_memory,
            "requires_rag": self.requires_rag,
            "requires_verification": self.requires_verification,
            "risk_level": self.risk_level,
            "expected_output": self.expected_output,
            "max_steps": self.max_steps,
            "timeout_seconds": self.timeout_seconds,
            "confirmation_required": self.confirmation_required,
        }



# Pattern matchers
UNSAFE_PATTERNS = [
    re.compile(r"\b(ddos|dos attack|syn flood|botnet|exploit payload)\b", re.IGNORECASE),
    re.compile(r"\b(phishing email|keylogger|malware script|credential harvest|ransomware)\b", re.IGNORECASE),
    re.compile(r"\b(rm\s+-rf\s+[/~]|drop\s+database|format\s+c:)\b", re.IGNORECASE),
]

MATH_EXACT_PATTERNS = [
    re.compile(r"^\s*(\d+(\.\d+)?\s*[\+\-\*\/\^%]\s*)+\d+(\.\d+)?\s*\??\s*$"),
    re.compile(r"^\s*what\s+is\s+([0-9\.\s\+\-\*\/\^\(\)%]+)\??\s*$", re.IGNORECASE),
    re.compile(r"^\s*calculate\s+([0-9\.\s\+\-\*\/\^\(\)%]+)\??\s*$", re.IGNORECASE),
    re.compile(r"^\s*(\d+)%\s+of\s+(\d+)\??\s*$", re.IGNORECASE),
]

CONVERSATION_GREETINGS = [
    re.compile(r"^(hi|hello|hey|greetings|good\s+(morning|afternoon|evening)|sup)\b", re.IGNORECASE),
    re.compile(r"^(how\s+are\s+you|how's\s+it\s+going|what's\s+up|thank\s+you|thanks)\b", re.IGNORECASE),
    re.compile(r"^(who\s+are\s+you|what\s+is\s+your\s+name)\??$", re.IGNORECASE),
]

MEMORY_WRITE_PATTERNS = [
    re.compile(r"\b(remember\s+(that)?|my\s+(favorite|preferred|name|email|budget|setting)\s+is)\b", re.IGNORECASE),
    re.compile(r"\b(save\s+(this\s+to\s+)?memory|store\s+(this\s+preference)?)\b", re.IGNORECASE),
    re.compile(r"\b(i\s+prefer\s+[\w\s]+|keep\s+in\s+mind\s+that)\b", re.IGNORECASE),
]

MEMORY_READ_PATTERNS = [
    re.compile(r"\b(what\s+is\s+my\s+(favorite|preferred|budget|name))\b", re.IGNORECASE),
    re.compile(r"\b(what\s+did\s+i\s+(say|tell\s+you)|do\s+you\s+remember\s+my)\b", re.IGNORECASE),
]

DOCUMENT_PATTERNS = [
    re.compile(r"\b(in\s+this\s+(document|pdf|file|context|article))\b", re.IGNORECASE),
    re.compile(r"\b(according\s+to\s+the\s+(document|pdf|file|context|text))\b", re.IGNORECASE),
    re.compile(r"\b(summarize\s+(this|the)\s+(document|pdf|file))\b", re.IGNORECASE),
]

PLANNING_PATTERNS = [
    re.compile(r"\b(plan\s+(my|a)|step-by-step\s+plan|break\s+down\s+the\s+steps|roadmap)\b", re.IGNORECASE),
    re.compile(r"\b(create\s+a\s+migration\s+plan|study\s+schedule)\b", re.IGNORECASE),
]

CODING_PATTERNS = [
    re.compile(r"\b(write|create|implement|debug|refactor)\s+(a\s+)?(python|javascript|typescript|rust|c\+\+|sql|html|css|function|class|algorithm|script)\b", re.IGNORECASE),
    re.compile(r"\b(time\s+complexity|space\s+complexity|big\s*-?\s*o|palindrome\s+function)\b", re.IGNORECASE),
]

MULTI_STEP_PATTERNS = [
    re.compile(r"\b(first\s+[\w\s,]+then\s+[\w\s,]+and\s+finally)\b", re.IGNORECASE),
    re.compile(r"\b(search\s+[\w\s]+and\s+summarize|calculate\s+[\w\s]+and\s+then\s+divide)\b", re.IGNORECASE),
]


def classify_intent(query: str, context: dict[str, Any] | None = None) -> IntentResult:
    """Classifies user query intent using high-precision deterministic rules and heuristics."""
    text = (query or "").strip()
    if not text:
        return IntentResult(
            intent=IntentCategory.UNKNOWN,
            confidence=1.0,
            requires_tool=False,
            requires_memory=False,
            requires_rag=False,
            requires_reasoning=False,
            requires_confirmation=False,
            safety_level="SAFE",
            reasoning="Empty input prompt",
        )

    # 1. Check Safety
    for pat in UNSAFE_PATTERNS:
        if pat.search(text):
            return IntentResult(
                intent=IntentCategory.UNSAFE_REQUEST,
                confidence=1.0,
                requires_tool=False,
                requires_memory=False,
                requires_rag=False,
                requires_reasoning=False,
                requires_confirmation=False,
                safety_level="UNSAFE",
                reasoning="Detected potentially harmful cyber-attack or system destruction pattern",
            )

    # 2. Check Mathematics
    for pat in MATH_EXACT_PATTERNS:
        if pat.search(text):
            return IntentResult(
                intent=IntentCategory.MATHEMATICS,
                confidence=0.98,
                requires_tool=True,
                requires_memory=False,
                requires_rag=False,
                requires_reasoning=True,
                requires_confirmation=False,
                safety_level="SAFE",
                target_tool="calculator",
                reasoning="Deterministic arithmetic calculation pattern",
            )

    # 3. Check Memory Write / Read
    for pat in MEMORY_WRITE_PATTERNS:
        if pat.search(text):
            return IntentResult(
                intent=IntentCategory.MEMORY,
                confidence=0.95,
                requires_tool=True,
                requires_memory=True,
                requires_rag=False,
                requires_reasoning=False,
                requires_confirmation=False,
                safety_level="SAFE",
                target_tool="memory_write",
                reasoning="Explicit user preference or memory store directive",
            )

    for pat in MEMORY_READ_PATTERNS:
        if pat.search(text):
            return IntentResult(
                intent=IntentCategory.MEMORY,
                confidence=0.95,
                requires_tool=True,
                requires_memory=True,
                requires_rag=False,
                requires_reasoning=False,
                requires_confirmation=False,
                safety_level="SAFE",
                target_tool="memory_read",
                reasoning="User preference or historical fact recall request",
            )

    # 4. Check Document / RAG Query
    for pat in DOCUMENT_PATTERNS:
        if pat.search(text):
            return IntentResult(
                intent=IntentCategory.DOCUMENT_QUESTION,
                confidence=0.92,
                requires_tool=True,
                requires_memory=False,
                requires_rag=True,
                requires_reasoning=True,
                requires_confirmation=False,
                safety_level="SAFE",
                target_tool="document_retrieval",
                reasoning="Reference to document, PDF, or context chunk",
            )

    # 4.5. Check Ambiguous conversion requests
    if re.match(r"^\s*convert\s+\d+(\.\d+)?\s*\.?\s*$", text, re.IGNORECASE):
        return IntentResult(
            intent=IntentCategory.CLARIFICATION_REQUIRED,
            confidence=0.98,
            requires_tool=False,
            requires_memory=False,
            requires_rag=False,
            requires_reasoning=False,
            requires_confirmation=False,
            safety_level="SAFE",
            reasoning="Ambiguous conversion request lacking source and target units",
        )

    # 5. Check Multi-step Task
    for pat in MULTI_STEP_PATTERNS:
        if pat.search(text):
            return IntentResult(
                intent=IntentCategory.MULTI_STEP_TASK,
                confidence=0.90,
                requires_tool=True,
                requires_memory=False,
                requires_rag=False,
                requires_reasoning=True,
                requires_confirmation=False,
                safety_level="SAFE",
                reasoning="Sequential task requiring multi-step execution",
            )

    # 6. Check Planning
    for pat in PLANNING_PATTERNS:
        if pat.search(text):
            return IntentResult(
                intent=IntentCategory.PLANNING,
                confidence=0.90,
                requires_tool=True,
                requires_memory=False,
                requires_rag=False,
                requires_reasoning=True,
                requires_confirmation=False,
                safety_level="SAFE",
                target_tool="structured_planning",
                reasoning="Task decomposition and roadmap planning request",
            )

    # 7. Check Coding
    for pat in CODING_PATTERNS:
        if pat.search(text):
            return IntentResult(
                intent=IntentCategory.CODING,
                confidence=0.88,
                requires_tool=False,
                requires_memory=False,
                requires_rag=False,
                requires_reasoning=True,
                requires_confirmation=False,
                safety_level="SAFE",
                reasoning="Programming, algorithm or software engineering query",
            )

    # 8. Check Conversation
    for pat in CONVERSATION_GREETINGS:
        if pat.search(text):
            return IntentResult(
                intent=IntentCategory.CONVERSATION,
                confidence=0.95,
                requires_tool=False,
                requires_memory=False,
                requires_rag=False,
                requires_reasoning=False,
                requires_confirmation=False,
                safety_level="SAFE",
                reasoning="Social greeting, gratitude, or persona question",
            )

    # 9. Factual QA Default
    if re.search(r"^(what|when|where|who|which|why|how)\b", text, re.IGNORECASE):
        return IntentResult(
            intent=IntentCategory.FACTUAL_QUESTION,
            confidence=0.80,
            requires_tool=False,
            requires_memory=False,
            requires_rag=False,
            requires_reasoning=False,
            requires_confirmation=False,
            safety_level="SAFE",
            reasoning="Informational query seeking factual knowledge",
        )

    # 10. General Conversation
    return IntentResult(
        intent=IntentCategory.CONVERSATION,
        confidence=0.70,
        requires_tool=False,
        requires_memory=False,
        requires_rag=False,
        requires_reasoning=False,
        requires_confirmation=False,
        safety_level="SAFE",
        reasoning="General conversational discourse",
    )


def build_task_spec(user_message: str, intent_result: IntentResult | None = None) -> TaskSpec:
    """Builds a structured internal TaskSpec from user message and intent classification."""
    if intent_result is None:
        intent_result = classify_intent(user_message)

    complexity = "LOW"
    if intent_result.intent in (IntentCategory.MULTI_STEP_TASK, IntentCategory.PLANNING):
        complexity = "HIGH"
    elif intent_result.intent in (IntentCategory.CODING, IntentCategory.REASONING, IntentCategory.MATHEMATICS):
        complexity = "MEDIUM"

    risk_level = "LOW"
    if intent_result.safety_level == "UNSAFE":
        risk_level = "HIGH"
    elif intent_result.requires_confirmation:
        risk_level = "MEDIUM"

    return TaskSpec(
        intent=intent_result.intent,
        complexity=complexity,
        requires_model=intent_result.intent not in (IntentCategory.MATHEMATICS, IntentCategory.CLARIFICATION_REQUIRED),
        requires_tool=intent_result.requires_tool,
        requires_memory=intent_result.requires_memory,
        requires_rag=intent_result.requires_rag,
        requires_verification=intent_result.intent in (IntentCategory.MATHEMATICS, IntentCategory.CODING, IntentCategory.DOCUMENT_QUESTION),
        risk_level=risk_level,
        expected_output="Direct answer" if complexity == "LOW" else "Structured response",
        max_steps=8 if complexity == "HIGH" else 4,
        timeout_seconds=30.0,
        confirmation_required=intent_result.requires_confirmation,
    )
