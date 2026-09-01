"""Self-verification and anti-hallucination layer with self-correction support."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any


from enum import Enum


class GroundingStatus(str, Enum):
    GROUNDED = "GROUNDED"
    PARTIALLY_GROUNDED = "PARTIALLY_GROUNDED"
    INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"
    CONTRADICTED = "CONTRADICTED"


@dataclass
class VerificationOutcome:
    verified: bool
    category: str
    confidence: float
    reason: str
    suggested_correction: str | None = None
    grounding_status: GroundingStatus | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "verified": self.verified,
            "category": self.category,
            "confidence": self.confidence,
            "reason": self.reason,
            "suggested_correction": self.suggested_correction,
            "grounding_status": self.grounding_status.value if self.grounding_status else None,
        }


def verify_json(text: str, required_keys: list[str] | None = None) -> VerificationOutcome:
    """Verifies that text is valid JSON and contains required keys."""
    # Find JSON block or extract directly
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    candidate = m.group(1).strip() if m else text.strip()

    try:
        data = json.loads(candidate)
        if required_keys and isinstance(data, dict):
            missing = [k for k in required_keys if k not in data]
            if missing:
                return VerificationOutcome(
                    verified=False,
                    category="json",
                    confidence=1.0,
                    reason=f"Missing required JSON keys: {missing}",
                )
        return VerificationOutcome(verified=True, category="json", confidence=1.0, reason="Valid JSON payload.")
    except Exception as e:
        return VerificationOutcome(
            verified=False,
            category="json",
            confidence=1.0,
            reason=f"Invalid JSON format: {str(e)}",
        )


def verify_arithmetic(prompt: str, response: str) -> VerificationOutcome:
    """Verifies that mathematical statements in the response match deterministic arithmetic."""
    # Find arithmetic expression in prompt or response, e.g. 45 * 12
    m = re.search(r"(\d+(\.\d+)?)\s*([\+\-\*\/])\s*(\d+(\.\d+)?)", prompt)
    if not m:
        m = re.search(r"(\d+(\.\d+)?)\s*([\+\-\*\/])\s*(\d+(\.\d+)?)", response)

    if m:
        try:
            a = float(m.group(1))
            op = m.group(3)
            b = float(m.group(4))

            if op == "+":
                expected = a + b
            elif op == "-":
                expected = a - b
            elif op == "*":
                expected = a * b
            elif op == "/" and b != 0:
                expected = a / b
            else:
                expected = None

            if expected is not None:
                if expected.is_integer():
                    expected_str = str(int(expected))
                else:
                    expected_str = str(round(expected, 4))

                # Check if expected number appears in response
                if expected_str in response or f"{expected:,.0f}" in response:
                    return VerificationOutcome(
                        verified=True,
                        category="arithmetic",
                        confidence=1.0,
                        reason=f"Calculated {a} {op} {b} = {expected_str}, found in response.",
                    )
                else:
                    return VerificationOutcome(
                        verified=False,
                        category="arithmetic",
                        confidence=0.95,
                        reason=f"Calculated {a} {op} {b} = {expected_str}, missing from response.",
                        suggested_correction=f"{expected_str}",
                    )
        except Exception:
            pass

    return VerificationOutcome(
        verified=True,
        category="arithmetic",
        confidence=0.8,
        reason="No simple arithmetic expression to verify.",
    )


def verify_code_syntax(code: str, language: str = "python") -> VerificationOutcome:
    """Verifies code syntax and delimiter balance."""
    lang = language.lower().strip()
    if lang == "python":
        try:
            ast.parse(code)
            return VerificationOutcome(verified=True, category="code_syntax", confidence=1.0, reason="Valid Python AST.")
        except SyntaxError as e:
            return VerificationOutcome(
                verified=False,
                category="code_syntax",
                confidence=1.0,
                reason=f"Python syntax error on line {e.lineno}: {e.msg}",
                suggested_correction="Fix indentation and syntax.",
            )

    # Balanced bracket check
    stack: list[str] = []
    matching = {")": "(", "}": "{", "]": "["}
    for char in code:
        if char in "({[":
            stack.append(char)
        elif char in ")}]":
            if not stack or stack[-1] != matching[char]:
                return VerificationOutcome(
                    verified=False,
                    category="code_syntax",
                    confidence=0.95,
                    reason=f"Unbalanced delimiter: '{char}'",
                )
            stack.pop()

    if stack:
        return VerificationOutcome(
            verified=False,
            category="code_syntax",
            confidence=0.95,
            reason=f"Unclosed delimiter: '{stack[-1]}'",
        )

    return VerificationOutcome(verified=True, category="code_syntax", confidence=1.0, reason="Balanced delimiters.")


def verify_document_grounding(context: str, response: str) -> VerificationOutcome:
    """Verifies that facts in response are grounded in the provided document context."""
    if not context.strip():
        return VerificationOutcome(verified=True, category="grounding", confidence=0.8, reason="No context provided.")

    # Check for honest abstention
    abstention_terms = [
        "not mentioned",
        "does not mention",
        "does not contain",
        "not specified",
        "cannot determine",
        "no information",
        "unknown",
    ]
    if any(term in response.lower() for term in abstention_terms):
        return VerificationOutcome(
            verified=True,
            category="grounding",
            confidence=0.95,
            reason="Response honestly recognizes absent information.",
        )

    # Word overlap check
    context_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", context.lower()))
    response_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", response.lower()))

    if not response_words:
        return VerificationOutcome(verified=True, category="grounding", confidence=0.8, reason="Response too short.")

    overlap = len(context_words.intersection(response_words))
    ratio = overlap / len(response_words)

    if ratio >= 0.20:
        return VerificationOutcome(
            verified=True,
            category="grounding",
            confidence=min(1.0, 0.7 + ratio),
            reason=f"Context grounding overlap ratio: {ratio:.1%}",
        )

    return VerificationOutcome(
        verified=False,
        category="grounding",
        confidence=0.75,
        reason=f"Low document grounding overlap ({ratio:.1%}).",
    )
