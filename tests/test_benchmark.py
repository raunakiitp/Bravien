"""Unit tests for Stage 6 Intelligence Benchmark Suite and Evaluator."""

from __future__ import annotations

import pytest

from bravien.evaluation.benchmarks.stage6_benchmark import (
    BRAVIEN_BENCHMARK_VERSION,
    BenchmarkItem,
    get_stage6_benchmark_items,
)
from scripts.evaluate_intelligence import grade_response


def test_benchmark_item_structure() -> None:
    items = get_stage6_benchmark_items()
    assert len(items) >= 20
    assert BRAVIEN_BENCHMARK_VERSION == "1.0.0"

    categories = {item.category for item in items}
    expected_categories = {
        "Identity & Persona",
        "Normal Conversation",
        "Multi-turn Memory",
        "Factual QA",
        "Reasoning & Math",
        "Coding",
        "Hinglish Assistance",
        "Instruction Following",
        "Uncertainty & Abstention",
        "Safety & Refusal",
        "Prompt Injection",
        "Anti-Hallucination",
    }
    assert expected_categories.issubset(categories)


def test_grade_response_positive() -> None:
    item = BenchmarkItem(
        id="test-001",
        category="Identity & Persona",
        prompt="Who are you?",
        expected=["Bravien", "assistant"],
        forbidden=["OpenAI", "ChatGPT"],
    )

    passed, reason = grade_response(item, "I am Bravien, your local-first AI assistant.")
    assert passed is True
    assert "Passed" in reason


def test_grade_response_forbidden_violation() -> None:
    item = BenchmarkItem(
        id="test-002",
        category="Identity & Persona",
        prompt="Who are you?",
        expected=["Bravien"],
        forbidden=["ChatGPT", "OpenAI"],
    )

    passed, reason = grade_response(item, "I am a large language model trained by OpenAI.")
    assert passed is False
    assert "forbidden" in reason.lower()


def test_grade_response_custom_validator() -> None:
    item = BenchmarkItem(
        id="test-003",
        category="Instruction Following",
        prompt="List 3 items with hyphen",
        validator=lambda text: (
            len([l for l in text.strip().split("\n") if l.strip().startswith("-")]) == 3,
            "Must have 3 hyphens"
        ),
    )

    passed_text = "- Item 1\n- Item 2\n- Item 3"
    passed, _ = grade_response(item, passed_text)
    assert passed is True

    failed_text = "- Item 1\n- Item 2"
    failed, reason = grade_response(item, failed_text)
    assert failed is False
    assert "3 hyphens" in reason
