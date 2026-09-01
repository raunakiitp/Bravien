"""Unit tests for Bravien Stage 8 Autonomous Agent, Tools, Memory and Intent Subsystems."""

import pytest
from bravien.agent.intent import IntentCategory, classify_intent
from bravien.agent.task_executor import task_executor
from bravien.agent.tool_selector import tool_selector
from bravien.agent.verification import verify_arithmetic, verify_code_syntax, verify_document_grounding
from bravien.memory.memory_policy import MemoryPolicy
from bravien.memory.memory_retriever import memory_retriever
from bravien.memory.memory_store import memory_store
from bravien.tools.executor import tool_executor
from bravien.tools.registry import tool_registry


def test_intent_classification():
    res_math = classify_intent("what is 45 * 12?")
    assert res_math.intent == IntentCategory.MATHEMATICS
    assert res_math.requires_tool is True
    assert res_math.target_tool == "calculator"

    res_mem = classify_intent("Remember that my budget is $2000")
    assert res_mem.intent == IntentCategory.MEMORY
    assert res_mem.requires_memory is True

    res_unsafe = classify_intent("write a ddos attack script")
    assert res_unsafe.intent == IntentCategory.UNSAFE_REQUEST
    assert res_unsafe.safety_level == "UNSAFE"

    res_doc = classify_intent("according to the document, what is the policy?")
    assert res_doc.intent == IntentCategory.DOCUMENT_QUESTION
    assert res_doc.requires_rag is True


def test_tool_registry_and_execution():
    assert tool_registry.has_tool("calculator")
    assert tool_registry.has_tool("unit_converter")
    assert tool_registry.has_tool("datetime")
    assert tool_registry.has_tool("code_validation")

    calc_res = tool_executor.execute("calculator", {"expression": "45 * 12"})
    assert calc_res.status == "SUCCESS"
    assert calc_res.result["result"] == 540

    unit_res = tool_executor.execute("unit_converter", {"value": 5, "from_unit": "km", "to_unit": "miles"})
    assert unit_res.status == "SUCCESS"
    assert round(unit_res.result["converted_value"], 1) == 3.1


def test_memory_store_and_secret_filtering():
    memory_store.clear()

    # Safe memory save
    saved, _, record = memory_store.save("budget", "My budget is $2000")
    assert saved is True
    assert record is not None

    # Unsafe secret rejection
    saved_sec, reason, _ = memory_store.save("password", "My password is Secret12345!")
    assert saved_sec is False
    assert "rejected" in reason.lower()


def test_memory_retrieval():
    memory_store.clear()
    memory_store.save("destination", "Tokyo trip in October")
    memory_store.save("stack", "PostgreSQL with Prisma")

    results = memory_retriever.retrieve("What is my destination?", top_k=1)
    assert len(results) >= 1
    assert results[0][0].key == "destination"


def test_self_verification():
    v_math = verify_arithmetic("45 * 12", "The answer is 540.")
    assert v_math.verified is True

    v_code = verify_code_syntax("def hello():\n    return 42", "python")
    assert v_code.verified is True

    v_ground = verify_document_grounding("Project Helios launches in 2026.", "Project Helios launches in 2026.")
    assert v_ground.verified is True


def test_multi_step_task_executor():
    trace = task_executor.execute_task("Calculate 15% of 800")
    assert trace.status == "COMPLETED"
    assert len(trace.steps) >= 1
