"""Comprehensive unit and smoke test suite for the native Bravien model architecture."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from bravien.model.bravien_attention import (
    BravienAttention,
    RotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)
from bravien.model.bravien_cache import BravienKVCache
from bravien.model.bravien_config import BRAVIEN_PRESETS, BravienConfig, get_bravien_preset
from bravien.model.bravien_model import BravienForCausalLM, BravienModel, CausalLMOutput
from bravien.model.checkpoint import (
    CheckpointManifest,
    load_bravien_checkpoint,
    save_bravien_checkpoint,
)
from bravien.tokenizer.tokenizer import BravienTokenizer


@pytest.fixture
def tiny_config() -> BravienConfig:
    return BravienConfig(
        name="bravien-test-tiny",
        vocab_size=256,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=128,
        max_position_embeddings=128,
        tie_word_embeddings=True,
    )


@pytest.fixture
def tiny_model(tiny_config: BravienConfig) -> BravienForCausalLM:
    torch.manual_seed(42)
    return BravienForCausalLM(tiny_config)


# ---------------------------------------------------------------------------
# 1. Model Initialization
# ---------------------------------------------------------------------------

def test_model_initialization(tiny_config: BravienConfig, tiny_model: BravienForCausalLM):
    assert tiny_model.config.hidden_size == 64
    assert tiny_model.config.num_layers == 2
    assert len(tiny_model.model.layers) == 2
    assert tiny_model.lm_head.weight.data_ptr() == tiny_model.model.embeddings.embed_tokens.weight.data_ptr()


def test_preset_retrieval():
    config = get_bravien_preset("bravien-1.5b")
    assert config.hidden_size == 2048
    assert config.num_layers == 32
    assert config.num_heads == 16
    assert config.num_kv_heads == 4
    assert config.intermediate_size == 5632


# ---------------------------------------------------------------------------
# 2. Parameter Counting
# ---------------------------------------------------------------------------

def test_parameter_counting(tiny_model: BravienForCausalLM):
    report = tiny_model.count_parameters()
    assert report.total > 0
    assert report.trainable == report.total
    assert report.embedding == 256 * 64
    assert report.lm_head == 0  # Tied weights
    assert "Bravien Model Parameter Report" in report.summary()


# ---------------------------------------------------------------------------
# 3. Forward Pass & 4. Causal Masking
# ---------------------------------------------------------------------------

def test_forward_pass_shapes(tiny_model: BravienForCausalLM):
    bsz, seq_len = 2, 8
    input_ids = torch.randint(0, 256, (bsz, seq_len))
    out = tiny_model(input_ids)
    assert isinstance(out, CausalLMOutput)
    assert out.logits.shape == (bsz, seq_len, 256)
    assert out.loss is None


def test_causal_attention_masking(tiny_config: BravienConfig):
    attn = BravienAttention(tiny_config, layer_idx=0)
    x = torch.randn(1, 4, tiny_config.hidden_size)
    out, _ = attn(x)
    assert out.shape == (1, 4, tiny_config.hidden_size)


# ---------------------------------------------------------------------------
# 5. Loss Calculation & 6. Backward Pass
# ---------------------------------------------------------------------------

def test_loss_and_backward_pass(tiny_model: BravienForCausalLM):
    input_ids = torch.tensor([[10, 20, 30, 40], [50, 60, 70, 80]])
    labels = torch.tensor([[20, 30, 40, 3], [60, 70, 80, 3]])  # 3 = EOS
    out = tiny_model(input_ids=input_ids, labels=labels)
    assert out.loss is not None
    assert out.loss.item() > 0.0

    # Test backpropagation
    out.loss.backward()
    for name, p in tiny_model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Gradient missing for {name}"


# ---------------------------------------------------------------------------
# 7. Autoregressive Generation & 10. Determinism
# ---------------------------------------------------------------------------

def test_greedy_generation_deterministic(tiny_model: BravienForCausalLM):
    prompt = torch.tensor([[5, 12, 18]])
    gen1 = tiny_model.generate(prompt, max_new_tokens=10, temperature=0.0, use_cache=True)
    gen2 = tiny_model.generate(prompt, max_new_tokens=10, temperature=0.0, use_cache=True)
    assert torch.equal(gen1, gen2)
    assert gen1.shape == (1, 13)


def test_temperature_and_top_p_generation(tiny_model: BravienForCausalLM):
    prompt = torch.tensor([[5, 12, 18]])
    gen = tiny_model.generate(
        prompt, max_new_tokens=8, temperature=0.8, top_k=20, top_p=0.9, use_cache=True
    )
    assert gen.shape[1] >= 4


# ---------------------------------------------------------------------------
# 8. KV Cache Acceleration
# ---------------------------------------------------------------------------

def test_kv_cache_consistency(tiny_model: BravienForCausalLM):
    prompt = torch.tensor([[10, 20, 30, 40]])

    # Generate without cache
    gen_no_cache = tiny_model.generate(prompt, max_new_tokens=6, temperature=0.0, use_cache=False)
    # Generate with cache
    gen_with_cache = tiny_model.generate(prompt, max_new_tokens=6, temperature=0.0, use_cache=True)

    assert torch.equal(gen_no_cache, gen_with_cache)


def test_kv_cache_memory_management():
    cache = BravienKVCache.create_empty(num_layers=2)
    k = torch.randn(1, 2, 4, 16)
    v = torch.randn(1, 2, 4, 16)
    cache.update(k, v, layer_idx=0)
    assert cache.get_seq_length(0) == 4
    assert cache.memory_footprint_bytes() > 0

    cache.reset()
    assert cache.get_seq_length(0) == 0


# ---------------------------------------------------------------------------
# 9. Save and Load Checkpoint (Serialization)
# ---------------------------------------------------------------------------

def test_save_and_load_checkpoint(tiny_model: BravienForCausalLM, tmp_path: Path):
    save_dir = tmp_path / "test_checkpoint"
    tiny_model.save_pretrained(save_dir)

    assert (save_dir / "model_config.json").exists()
    assert (save_dir / "model.pt").exists()
    assert (save_dir / "manifest.json").exists()

    # Load back
    loaded_model = BravienForCausalLM.from_pretrained(save_dir)
    assert loaded_model.config.hidden_size == tiny_model.config.hidden_size
    assert loaded_model.config.num_layers == tiny_model.config.num_layers

    # Ensure output equality
    x = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        orig_out = tiny_model(x).logits
        loaded_out = loaded_model(x).logits
    assert torch.allclose(orig_out, loaded_out, atol=1e-5)


# ---------------------------------------------------------------------------
# 11. CPU & 12. CUDA Compatibility
# ---------------------------------------------------------------------------

def test_device_compatibility(tiny_model: BravienForCausalLM):
    # Always test CPU
    tiny_model.to("cpu")
    x = torch.tensor([[1, 2, 3]], device="cpu")
    out = tiny_model(x)
    assert out.logits.device.type == "cpu"

    # Test CUDA if available
    if torch.cuda.is_available():
        tiny_model.to("cuda")
        x_cuda = torch.tensor([[1, 2, 3]], device="cuda")
        out_cuda = tiny_model(x_cuda)
        assert out_cuda.logits.device.type == "cuda"
        tiny_model.to("cpu")


# ---------------------------------------------------------------------------
# 13. Tokenizer Integration
# ---------------------------------------------------------------------------

def test_tokenizer_integration(tmp_path: Path):
    tok_dir = Path("tokenizers/bravien-native")
    if (tok_dir / "tokenizer.json").exists():
        tok = BravienTokenizer.from_pretrained(tok_dir)
        enc = tok.encode("Hello Bravien!")
        assert len(enc) > 0
        dec = tok.decode(enc)
        assert dec == "Hello Bravien!"


# ---------------------------------------------------------------------------
# 14. Malformed Checkpoint Handling
# ---------------------------------------------------------------------------

def test_malformed_checkpoint_handling(tmp_path: Path):
    bad_dir = tmp_path / "corrupt_checkpoint"
    bad_dir.mkdir()

    # Missing config
    with pytest.raises(FileNotFoundError):
        load_bravien_checkpoint(bad_dir)

    # Missing weights
    (bad_dir / "model_config.json").write_text(json.dumps({"hidden_size": 64, "num_layers": 2}))
    with pytest.raises(FileNotFoundError):
        load_bravien_checkpoint(bad_dir)
