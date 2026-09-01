"""Unit tests for Bravien Stage 7 benchmark and custom validators."""

import pytest
from bravien.evaluation.benchmarks.stage7_benchmark import (
    BRAVIEN_STAGE7_BENCHMARK_VERSION,
    BenchmarkItem,
    get_stage7_benchmark_items,
)
from scripts.evaluate_intelligence import grade_response


def test_stage7_benchmark_items_count_and_schema():
    items = get_stage7_benchmark_items()
    assert len(items) >= 30
    assert BRAVIEN_STAGE7_BENCHMARK_VERSION == "2.0.0"

    categories = set(item.category for item in items)
    assert "Identity & Persona" in categories
    assert "Normal Conversation" in categories
    assert "Multi-turn Memory" in categories
    assert "Factual QA" in categories
    assert "Reasoning & Math" in categories
    assert "Coding" in categories
    assert "Hinglish Assistance" in categories
    assert "Instruction Following" in categories
    assert "Uncertainty & Abstention" in categories
    assert "Safety & Ethical Refusal" in categories
    assert "Prompt Injection Defense" in categories
    assert "Anti-Hallucination" in categories
    assert "Tool Selection" in categories
    assert "Tool Execution" in categories
    assert "RAG Grounding" in categories
    assert "Task Planning" in categories
    assert "Multi-step Execution" in categories
    assert "Self-Correction" in categories


def test_custom_validators():
    items = {item.id: item for item in get_stage7_benchmark_items()}

    # Test python code validator
    code_item = items["code-001"]
    passed, _ = grade_response(code_item, "def is_palindrome(s: str) -> bool:\n    return s == s[::-1]")
    assert passed is True

    # Test json validator
    json_item = items["inst-002"]
    passed, _ = grade_response(json_item, '{"status": "ok", "code": 200}')
    assert passed is True

    # Test bullet list validator
    inst_item = items["inst-001"]
    passed, _ = grade_response(inst_item, "- Faster feedback\n- Regression prevention\n- Better design")
    assert passed is True


def test_grading_logic_assertions():
    item = BenchmarkItem(
        id="test-001",
        category="Test",
        prompt="Sample",
        expected=["Bravien"],
        forbidden=["OpenAI"],
    )

    # Valid response
    passed, _ = grade_response(item, "I am Bravien.")
    assert passed is True

    # Forbidden violation
    passed, reason = grade_response(item, "I am an OpenAI model called Bravien.")
    assert passed is False
    assert "forbidden" in reason

    # Missing expected concept
    passed, reason = grade_response(item, "I am an assistant.")
    assert passed is False
    assert "Missing expected" in reason
