"""Comprehensive unit and integration tests for Bravien Data Pipeline (Stage 1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bravien.data.curation import get_curated_seed_examples
from bravien.data.deduplicate import Deduplicator
from bravien.data.filter import (
    calculate_quality_score,
    contains_secrets,
    detect_language,
)
from bravien.data.pipeline import BravienDataPipeline, PipelineConfig
from bravien.data.schema import (
    BravienMessage,
    BravienTrainingExample,
    ValidationError,
)
from bravien.data.synthetic import ProceduralCorpusGenerator


def test_schema_validation_valid():
    ex = BravienTrainingExample(
        id="test-1",
        source="test",
        category="coding",
        messages=[
            BravienMessage(role="system", content="System instruction"),
            BravienMessage(role="user", content="User prompt"),
            BravienMessage(role="assistant", content="Assistant reply"),
        ],
        quality_score=0.95,
        language="en",
    )
    assert ex.id == "test-1"
    assert ex.messages[0].role == "system"
    assert ex.messages[-1].role == "assistant"
    assert ex.estimate_tokens() > 0

    d = ex.to_dict()
    assert d["category"] == "coding"
    assert len(d["messages"]) == 3

    # Round trip
    reconstructed = BravienTrainingExample.from_dict(d)
    assert reconstructed.id == ex.id
    assert len(reconstructed.messages) == len(ex.messages)


def test_schema_rejects_invalid_roles():
    with pytest.raises(ValidationError):
        BravienMessage(role="invalid_role", content="text")  # type: ignore


def test_schema_rejects_empty_content():
    with pytest.raises(ValidationError):
        BravienMessage(role="user", content="   ")


def test_schema_rejects_missing_assistant_final_turn():
    with pytest.raises(ValidationError):
        BravienTrainingExample(
            id="test-bad-end",
            source="test",
            category="coding",
            messages=[
                BravienMessage(role="user", content="Hello"),
            ],
        )


def test_schema_rejects_missing_user_turn():
    with pytest.raises(ValidationError):
        BravienTrainingExample(
            id="test-no-user",
            source="test",
            category="coding",
            messages=[
                BravienMessage(role="system", content="Hello"),
                BravienMessage(role="assistant", content="Hello back"),
            ],
        )


def test_secret_detection_true_positives():
    # Private Key
    key_text = "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA0\n-----END RSA PRIVATE KEY-----"
    has_sec, sec_type = contains_secrets(key_text)
    assert has_sec is True
    assert sec_type == "private_key"

    # GitHub PAT
    gh_text = "ghp_1234567890abcdefghijklmnopqrstuvwxyz12"
    has_sec, sec_type = contains_secrets(gh_text)
    assert has_sec is True
    assert sec_type == "github_token"

    # AWS Access Key
    aws_text = "AKIAIOSFODNN7EXAMPLE"
    has_sec, sec_type = contains_secrets(aws_text)
    assert has_sec is True
    assert sec_type == "aws_access_key"


def test_secret_detection_true_negatives():
    safe_code = """
    def set_password(password: str):
        # API key is configured via environment
        api_key = os.environ.get("OPENAI_API_KEY")
        token = "<KEY_PLACEHOLDER>"
        return True
    """
    has_sec, _ = contains_secrets(safe_code)
    assert has_sec is False


def test_language_detection():
    assert detect_language("This is a standard English text explaining algorithms.") == "en"
    assert detect_language("Bravien, mujhe Python me list aur tuple ka difference samjhao please.") == "en-hi"


def test_quality_scoring():
    good_messages = [
        {"role": "user", "content": "Explain binary search in one paragraph."},
        {"role": "assistant", "content": "Binary search is an efficient algorithm for finding an item from a sorted list of items by repeatedly dividing in half the portion of the list that could contain the item."},
    ]
    score, reasons = calculate_quality_score(good_messages)
    assert score >= 0.8
    assert not reasons

    # Secret penalty
    bad_messages = [
        {"role": "user", "content": "Here is my secret"},
        {"role": "assistant", "content": "ghp_1234567890abcdefghijklmnopqrstuvwxyz12 is your token."},
    ]
    score, reasons = calculate_quality_score(bad_messages)
    assert score == 0.0
    assert any("secret" in r for r in reasons)


def test_curated_seeds_coverage():
    seeds = get_curated_seed_examples()
    assert len(seeds) >= 10
    categories = {s.category for s in seeds}
    assert "general_knowledge" in categories
    assert "instruction_following" in categories
    assert "dialogue" in categories
    assert "reasoning" in categories
    assert "coding" in categories
    assert "safety_refusal" in categories
    assert "bravien_assistant" in categories
    assert "tool_use" in categories
    assert "hinglish" in categories

    for s in seeds:
        s.validate()


def test_deduplicator_examples():
    ex1 = BravienTrainingExample(
        id="ex-1",
        source="s1",
        category="coding",
        messages=[
            BravienMessage(role="user", content="Write a function to add two numbers"),
            BravienMessage(role="assistant", content="def add(a, b): return a + b"),
        ],
    )
    ex2 = BravienTrainingExample(
        id="ex-2",
        source="s2",
        category="coding",
        messages=[
            BravienMessage(role="user", content="  write a function to add two numbers  "),
            BravienMessage(role="assistant", content="def add(a, b): return a + b"),
        ],
    )
    ex3 = BravienTrainingExample(
        id="ex-3",
        source="s3",
        category="coding",
        messages=[
            BravienMessage(role="user", content="Calculate square root of 16"),
            BravienMessage(role="assistant", content="The square root of 16 is 4."),
        ],
    )

    dedup = Deduplicator()
    kept = list(dedup.filter_examples([ex1, ex2, ex3]))
    assert len(kept) == 2
    assert kept[0].id == "ex-1"
    assert kept[1].id == "ex-3"


def test_pipeline_end_to_end(tmp_path: Path):
    out_dir = tmp_path / "processed"
    manifest_dir = tmp_path / "manifests"
    sample_dir = tmp_path / "samples"
    raw_file = tmp_path / "raw.jsonl"

    # Write test raw data
    with open(raw_file, "w", encoding="utf-8") as f:
        for i in range(30):
            record = {
                "instruction": f"Solve problem number {i}",
                "input": "",
                "output": f"The answer to problem number {i} is {i * 2}.",
            }
            f.write(json.dumps(record) + "\n")

    config = PipelineConfig(
        output_dir=out_dir,
        manifest_dir=manifest_dir,
        sample_dir=sample_dir,
        raw_sources=[raw_file],
        include_curated_seeds=True,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=100,
    )

    pipeline = BravienDataPipeline(config)
    manifest = pipeline.run()

    assert manifest["statistics"]["total_kept"] > 0
    assert (out_dir / "train.jsonl").exists()
    assert (out_dir / "val.jsonl").exists()
    assert (out_dir / "test.jsonl").exists()
    assert (manifest_dir / "dataset_manifest.json").exists()
    assert (sample_dir / "sample_conversations.jsonl").exists()

    # Check zero leakage between splits
    train_prompts = set()
    with open(out_dir / "train.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            ex = BravienTrainingExample.from_dict(json.loads(line))
            train_prompts.add(ex.user_prompt_fingerprint())

    val_prompts = set()
    with open(out_dir / "val.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            ex = BravienTrainingExample.from_dict(json.loads(line))
            val_prompts.add(ex.user_prompt_fingerprint())

    test_prompts = set()
    with open(out_dir / "test.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            ex = BravienTrainingExample.from_dict(json.loads(line))
            test_prompts.add(ex.user_prompt_fingerprint())

    assert train_prompts.isdisjoint(val_prompts)
    assert train_prompts.isdisjoint(test_prompts)
    assert val_prompts.isdisjoint(test_prompts)


def test_procedural_generator_domains():
    gen = ProceduralCorpusGenerator(seed=123)
    maths = list(gen.generate_math_reasoning(count=10))
    codes = list(gen.generate_coding_examples(count=10))
    tools = list(gen.generate_agent_and_tool_examples(count=10))
    safes = list(gen.generate_safety_and_uncertainty(count=10))
    hings = list(gen.generate_hinglish_examples(count=10))
    diags = list(gen.generate_dialogue_and_instruction(count=10))

    assert len(maths) == 10
    assert len(codes) == 10
    assert len(tools) == 10
    assert len(safes) == 10
    assert len(hings) == 10
    assert len(diags) == 10

    for ex in (maths + codes + tools + safes + hings + diags):
        ex.validate()
        assert ex.messages[0].role == "system"
        assert ex.messages[-1].role == "assistant"


def test_pipeline_scaling_and_balancing(tmp_path: Path):
    out_dir = tmp_path / "scaled_processed"
    manifest_dir = tmp_path / "scaled_manifests"
    sample_dir = tmp_path / "scaled_samples"

    config = PipelineConfig(
        output_dir=out_dir,
        manifest_dir=manifest_dir,
        sample_dir=sample_dir,
        include_curated_seeds=True,
        include_synthetic_scaling=True,
        target_samples=500,
        balance_categories=True,
        max_category_share=0.35,
        seed=42,
    )

    pipeline = BravienDataPipeline(config)
    manifest = pipeline.run()

    total_kept = manifest["statistics"]["total_kept"]
    assert total_kept >= 400

    # Ensure category balance: no category exceeds 35%
    cat_dist = manifest["statistics"]["category_distribution"]
    for cat, count in cat_dist.items():
        share = count / total_kept
        assert share <= 0.36, f"Category {cat} share {share:.2f} exceeded cap 0.35"

