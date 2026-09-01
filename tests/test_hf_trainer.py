"""Unit tests for Stage 3 HuggingFace SFT Trainer, dataset masking, and checkpointing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from bravien.training.hf_trainer import (
    HFTrainer,
    HFTrainingConfig,
    SFTDataset,
    sft_collate_fn,
)


def test_sft_dataset_loss_masking(tmp_path: Path):
    data_file = tmp_path / "test_train.jsonl"
    record = {
        "id": "ex-1",
        "messages": [
            {"role": "system", "content": "You are Bravien."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hello! How can I help you today?"},
        ],
    }
    with open(data_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset = SFTDataset(data_file, tokenizer, max_length=128)
    assert len(dataset) == 1

    item = dataset[0]
    input_ids = item["input_ids"]
    labels = item["labels"]
    attention_mask = item["attention_mask"]

    assert len(input_ids) == len(labels) == len(attention_mask)

    # Verify that system and user tokens are masked with -100
    assert labels[0].item() == -100  # <|im_start|>system

    # Verify that assistant tokens contain non-negative label ids
    unmasked_labels = labels[labels != -100]
    assert len(unmasked_labels) > 0


def test_sft_collate_fn(tmp_path: Path):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    batch_items = [
        {
            "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
            "labels": torch.tensor([-100, -100, 3], dtype=torch.long),
            "attention_mask": torch.tensor([1, 1, 1], dtype=torch.long),
        },
        {
            "input_ids": torch.tensor([1, 2, 3, 4, 5], dtype=torch.long),
            "labels": torch.tensor([-100, -100, -100, 4, 5], dtype=torch.long),
            "attention_mask": torch.tensor([1, 1, 1, 1, 1], dtype=torch.long),
        },
    ]

    collated = sft_collate_fn(batch_items, pad_token_id=0)

    assert collated["input_ids"].shape == (2, 5)
    assert collated["labels"].shape == (2, 5)
    assert collated["attention_mask"].shape == (2, 5)

    # First item must be padded at the end
    assert collated["input_ids"][0, 3].item() == 0
    assert collated["labels"][0, 3].item() == -100
    assert collated["attention_mask"][0, 3].item() == 0
