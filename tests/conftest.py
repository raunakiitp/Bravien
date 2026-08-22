"""Shared fixtures for the Bravien Python test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Tests run against the working tree, not an installed package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session")
def tiny_config():
    """A model small enough to train and run inside a unit test."""
    from bravien.model.config import BravienConfig

    return BravienConfig(
        vocab_size=128,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
        dropout=0.0,
        attention_dropout=0.0,
    )


@pytest.fixture
def tiny_model(tiny_config):
    """A deterministic, eval-mode model. Dropout is off so outputs are stable."""
    import torch

    from bravien.model.model import BravienForCausalLM

    torch.manual_seed(0)
    model = BravienForCausalLM(tiny_config)
    model.eval()
    return model


@pytest.fixture(scope="session")
def trained_tokenizer():
    """A real tokenizer, trained here rather than loaded from `tokenizers/`.

    Training takes about a second at this vocabulary size, and it keeps the suite
    honest: a test that depended on a checked-out artifact would pass or fail
    based on whether someone had run `train_tokenizer.py`, not on the code.
    """
    from bravien.data.download import generate_seed_corpus
    from bravien.tokenizer.train import train_tokenizer

    texts = [doc.text for doc in generate_seed_corpus(120, seed=0)]
    return train_tokenizer(texts, vocab_size=512, min_frequency=2)
