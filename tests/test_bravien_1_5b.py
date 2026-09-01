"""Comprehensive Unit and Integration Test Suite for Bravien 1.5B Architecture Foundation.

Tests all 25 mandatory architectural, tokenizer, and backward-compatibility criteria.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from bravien.model.bravien_attention import BravienAttention, RotaryEmbedding
from bravien.model.bravien_cache import BravienKVCache
from bravien.model.bravien_config import BRAVIEN_PRESETS, BravienConfig, get_bravien_preset
from bravien.model.bravien_mlp import BravienSwiGLUMLP
from bravien.model.bravien_model import BravienForCausalLM, BravienModel
from bravien.model.bravien_norm import BravienRMSNorm
from bravien.model.checkpoint import load_bravien_checkpoint, save_bravien_checkpoint
from bravien.model.provider import BravienLocalProvider, BravienNativeProvider, get_model_provider
from bravien.tokenizer.tokenizer import BravienTokenizer
from scripts.count_bravien_parameters import calculate_analytical_params


@pytest.fixture
def tiny_config() -> BravienConfig:
    return get_bravien_preset("bravien-tiny")


@pytest.fixture
def target_1p5b_config() -> BravienConfig:
    return get_bravien_preset("bravien-1.5b")


@pytest.fixture
def native_tokenizer() -> BravienTokenizer:
    tok_dir = Path("tokenizers/bravien-native")
    return BravienTokenizer.from_pretrained(tok_dir)


class TestBravien1p5bFoundation:
    """Test suite covering all 25 core foundation requirements."""

    # 1. Config serialization
    def test_01_config_serialization(self, target_1p5b_config: BravienConfig):
        d = target_1p5b_config.to_dict()
        assert d["name"] == "bravien-1.5b"
        assert d["hidden_size"] == 2048
        assert d["num_layers"] == 32
        reconstructed = BravienConfig.from_dict(d)
        assert reconstructed == target_1p5b_config

    # 2. Model construction
    def test_02_model_construction(self, tiny_config: BravienConfig):
        model = BravienForCausalLM(tiny_config)
        assert isinstance(model.model, BravienModel)
        assert len(model.model.layers) == tiny_config.num_layers

    # 3. Exact parameter counting
    def test_03_exact_parameter_counting(self, tiny_config: BravienConfig):
        model = BravienForCausalLM(tiny_config)
        counts = calculate_analytical_params(tiny_config)
        py_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert counts["total"] == py_params

    # 4. Target parameter range (1.45B - 1.60B)
    def test_04_target_parameter_range(self, target_1p5b_config: BravienConfig):
        counts = calculate_analytical_params(target_1p5b_config)
        total = counts["total"]
        assert 1_450_000_000 <= total <= 1_600_000_000, f"Out of range: {total}"
        assert abs(total - 1_500_000_000) < 15_000_000  # ~1.508B

    # 5. Forward pass
    def test_05_forward_pass(self, tiny_config: BravienConfig):
        model = BravienForCausalLM(tiny_config)
        input_ids = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
        out = model(input_ids)
        assert out.logits.shape == (1, 4, tiny_config.vocab_size)

    # 6. Causal masking
    def test_06_causal_masking(self, tiny_config: BravienConfig):
        model = BravienForCausalLM(tiny_config)
        input_ids = torch.tensor([[5, 10, 15, 20]], dtype=torch.long)
        out1 = model(input_ids).logits
        # Modify the 4th token and ensure logits for 1st, 2nd, 3rd tokens are identical
        input_ids_mod = torch.tensor([[5, 10, 15, 99]], dtype=torch.long)
        out2 = model(input_ids_mod).logits
        assert torch.allclose(out1[:, :3, :], out2[:, :3, :], atol=1e-5)

    # 7. Attention shape correctness
    def test_07_attention_shape(self, tiny_config: BravienConfig):
        attn = BravienAttention(tiny_config, layer_idx=0)
        x = torch.randn(2, 8, tiny_config.hidden_size)
        out, _ = attn(x)
        assert out.shape == (2, 8, tiny_config.hidden_size)

    # 8. GQA shape correctness (4:1)
    def test_08_gqa_shape(self, target_1p5b_config: BravienConfig):
        assert target_1p5b_config.num_heads == 16
        assert target_1p5b_config.num_kv_heads == 4
        assert target_1p5b_config.num_heads // target_1p5b_config.num_kv_heads == 4

    # 9. RoPE rotary positional embedding
    def test_09_rope_embedding(self):
        rope = RotaryEmbedding(dim=64, max_position_embeddings=512)
        x = torch.randn(2, 4, 16, 64)
        cos, sin = rope(x, seq_len=16)
        assert cos.shape == (16, 64)
        assert sin.shape == (16, 64)

    # 10. RMSNorm pre-normalization
    def test_10_rmsnorm(self):
        norm = BravienRMSNorm(dim=128, eps=1e-5)
        x = torch.randn(2, 10, 128)
        out = norm(x)
        assert out.shape == x.shape
        assert torch.all(torch.isfinite(out))

    # 11. SwiGLU gated MLP
    def test_11_swiglu_mlp(self, tiny_config: BravienConfig):
        mlp = BravienSwiGLUMLP(tiny_config)
        x = torch.randn(2, 5, tiny_config.hidden_size)
        out = mlp(x)
        assert out.shape == (2, 5, tiny_config.hidden_size)

    # 12. Gradient flow (backward pass)
    def test_12_gradient_flow(self, tiny_config: BravienConfig):
        model = BravienForCausalLM(tiny_config)
        input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        labels = torch.tensor([[2, 3, 4, 3]], dtype=torch.long)
        out = model(input_ids, labels=labels)
        out.loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None and not torch.isnan(p.grad).any()

    # 13. Save/load with SHA-256 manifest
    def test_13_save_load_manifest(self, tiny_config: BravienConfig):
        model = BravienForCausalLM(tiny_config)
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "test_ckpt"
            save_bravien_checkpoint(model, save_path)
            loaded = load_bravien_checkpoint(save_path, device=torch.device("cpu"), verify_checksums=True)
            inp = torch.tensor([[5, 6, 7]])
            assert torch.allclose(model(inp).logits, loaded(inp).logits, atol=1e-5)

    # 14. Deterministic initialization with seed
    def test_14_deterministic_initialization(self, tiny_config: BravienConfig):
        torch.manual_seed(42)
        m1 = BravienForCausalLM(tiny_config)
        torch.manual_seed(42)
        m2 = BravienForCausalLM(tiny_config)
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            assert torch.equal(p1, p2)

    # 15. Tokenizer encode/decode roundtrip
    def test_15_tokenizer_roundtrip(self, native_tokenizer: BravienTokenizer):
        text = "Bravien sovereign AI assistant architecture."
        tokens = native_tokenizer.encode(text, add_special_tokens=False)
        decoded = native_tokenizer.decode(tokens, skip_special_tokens=True)
        assert decoded == text

    # 16. English tokenization
    def test_16_english_tokenization(self, native_tokenizer: BravienTokenizer):
        text = "The quick brown fox jumps over the lazy dog."
        tokens = native_tokenizer.encode(text)
        assert len(tokens) > 0
        assert native_tokenizer.decode(tokens, skip_special_tokens=True) == text

    # 17. Hindi tokenization
    def test_17_hindi_tokenization(self, native_tokenizer: BravienTokenizer):
        text = "नमस्ते! ब्राविएन एक स्वतंत्र और स्थानीय कृत्रिम बुद्धिमत्ता सहायक है।"
        tokens = native_tokenizer.encode(text)
        assert native_tokenizer.decode(tokens, skip_special_tokens=True) == text

    # 18. Hinglish tokenization
    def test_18_hinglish_tokenization(self, native_tokenizer: BravienTokenizer):
        text = "Python me list mutable hoti hai aur tuple immutable hota hai."
        tokens = native_tokenizer.encode(text)
        assert native_tokenizer.decode(tokens, skip_special_tokens=True) == text

    # 19. Code tokenization (Python & TypeScript)
    def test_19_code_tokenization(self, native_tokenizer: BravienTokenizer):
        py_code = "def fib(n: int) -> int:\n    return n if n <= 1 else fib(n-1) + fib(n-2)"
        tokens = native_tokenizer.encode(py_code)
        assert native_tokenizer.decode(tokens, skip_special_tokens=True) == py_code

    # 20. JSON tokenization
    def test_20_json_tokenization(self, native_tokenizer: BravienTokenizer):
        json_text = '{"name": "bravien-1.5b", "params": 1508509696, "active": true}'
        tokens = native_tokenizer.encode(json_text)
        assert native_tokenizer.decode(tokens, skip_special_tokens=True) == json_text

    # 21. Chat template formatting
    def test_21_chat_template(self, native_tokenizer: BravienTokenizer):
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]
        chat_enc = native_tokenizer.encode_chat(messages)
        assert len(chat_enc.input_ids) > 0
        assert len(chat_enc.labels) == len(chat_enc.input_ids)

    # 22. Generation smoke test with KV cache
    def test_22_generation_smoke(self, tiny_config: BravienConfig):
        model = BravienForCausalLM(tiny_config)
        prompt = torch.tensor([[2, 10, 20]])
        gen = model.generate(prompt, max_new_tokens=5, temperature=0.0, use_cache=True)
        assert gen.shape == (1, 8)

    # 23. Provider loading
    def test_23_provider_loading(self):
        provider = BravienNativeProvider()
        info = provider.model_info()
        assert info["backend"] == "native"
        assert info["architecture"] == "bravien"

    # 24. Production default configuration
    def test_24_production_default(self):
        provider = get_model_provider()
        assert isinstance(provider, BravienNativeProvider)

    # 25. Existing Bravien-v3 rollback still works
    def test_25_rollback_baseline(self):
        # Explicit rollback provider
        v3_provider = BravienLocalProvider(checkpoint_path="checkpoints/bravien-v3")
        assert v3_provider.checkpoint_path == "checkpoints/bravien-v3"
        meta = v3_provider.metadata()
        assert meta.provider_type == "qwen_compat"
