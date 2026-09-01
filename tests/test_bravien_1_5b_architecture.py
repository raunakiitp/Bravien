"""Stage 10 Architectural Verification Test Suite for Bravien-1.5B."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from bravien.model.bravien_config import BRAVIEN_PRESETS, BravienConfig, get_bravien_preset
from bravien.model.bravien_model import BravienForCausalLM
from bravien.model.checkpoint import load_bravien_checkpoint, save_bravien_checkpoint
from bravien.model.factory import ModelFactory
from bravien.model.parameter_count import count_parameters
from bravien.model.provider import BravienLocalProvider, BravienNativeProvider, get_model_provider
from bravien.tokenizer.tokenizer import BravienTokenizer


@pytest.fixture
def cfg_1p5b() -> BravienConfig:
    return get_bravien_preset("bravien-1.5b")


@pytest.fixture
def cfg_tiny() -> BravienConfig:
    return get_bravien_preset("bravien-tiny")


class TestStage10Architecture:
    """Stage 10 mandatory architecture independence and parameter verification."""

    # 1. Config loads
    def test_01_config_loads(self, cfg_1p5b: BravienConfig):
        assert cfg_1p5b.name == "bravien-1.5b"
        assert cfg_1p5b.hidden_size == 2048
        assert cfg_1p5b.num_layers == 32

    # 2. Model initializes independently
    def test_02_model_initialization(self, cfg_tiny: BravienConfig):
        model = BravienForCausalLM(cfg_tiny)
        assert isinstance(model, BravienForCausalLM)
        assert hasattr(model, "model")
        assert hasattr(model, "lm_head")

    # 3. Parameter count is ~1.5B (1,508,509,696)
    def test_03_parameter_count_1p5b(self, cfg_1p5b: BravienConfig):
        with torch.device("meta"):
            model = BravienForCausalLM(cfg_1p5b)
            report = count_parameters(model)
        assert report["total_parameters"] == 1_508_509_696
        assert report["is_target_1p5b"] is True

    # 4. Parameter count is NOT ~494M
    def test_04_parameter_count_not_494m(self, cfg_1p5b: BravienConfig):
        with torch.device("meta"):
            model = BravienForCausalLM(cfg_1p5b)
            report = count_parameters(model)
        assert report["is_legacy_494m"] is False
        assert report["total_parameters"] > 1_000_000_000

    # 5. model_type is Bravien (Hard check: NOT qwen2)
    def test_05_model_type_sovereign(self, cfg_1p5b: BravienConfig):
        assert cfg_1p5b.model_type == "bravien"
        assert cfg_1p5b.model_type != "qwen2"
        assert cfg_1p5b.model_type != "qwen"

    # 6. architecture is BravienForCausalLM
    def test_06_architecture_identity(self, cfg_1p5b: BravienConfig):
        assert cfg_1p5b.architecture == "BravienForCausalLM"

    # 7. Forward pass works
    def test_07_forward_pass(self, cfg_tiny: BravienConfig):
        model = BravienForCausalLM(cfg_tiny)
        inp = torch.tensor([[10, 20, 30]])
        out = model(inp)
        assert out.logits.shape == (1, 3, cfg_tiny.vocab_size)

    # 8. Loss calculation works
    def test_08_loss_calculation(self, cfg_tiny: BravienConfig):
        model = BravienForCausalLM(cfg_tiny)
        inp = torch.tensor([[5, 10, 15, 20]])
        labels = torch.tensor([[10, 15, 20, 3]])
        out = model(inp, labels=labels)
        assert out.loss is not None
        assert not torch.isnan(out.loss)
        assert not torch.isinf(out.loss)

    # 9. Causal attention works
    def test_09_causal_attention(self, cfg_tiny: BravienConfig):
        model = BravienForCausalLM(cfg_tiny)
        inp1 = torch.tensor([[1, 2, 3, 4]])
        inp2 = torch.tensor([[1, 2, 3, 99]])
        out1 = model(inp1).logits
        out2 = model(inp2).logits
        assert torch.allclose(out1[:, :3, :], out2[:, :3, :], atol=1e-5)

    # 10. Generation works
    def test_10_generation(self, cfg_tiny: BravienConfig):
        model = BravienForCausalLM(cfg_tiny)
        prompt = torch.tensor([[2, 10]])
        gen = model.generate(prompt, max_new_tokens=4, temperature=0.0, use_cache=True)
        assert gen.shape == (1, 6)

    # 11. Save/load roundtrip works
    def test_11_save_load_roundtrip(self, cfg_tiny: BravienConfig):
        model = BravienForCausalLM(cfg_tiny)
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "ckpt"
            save_bravien_checkpoint(model, save_path)
            loaded = load_bravien_checkpoint(save_path, device=torch.device("cpu"))
            inp = torch.tensor([[4, 8, 12]])
            assert torch.allclose(model(inp).logits, loaded(inp).logits, atol=1e-5)

    # 12. ModelFactory resolution
    def test_12_model_factory(self):
        m = ModelFactory.create_model("bravien-tiny", device="cpu")
        assert isinstance(m, BravienForCausalLM)
        p = ModelFactory.create_provider("native")
        assert isinstance(p, BravienNativeProvider)

    # 13. Tokenizer loads and supports special tokens
    def test_13_tokenizer_loads(self):
        tok = BravienTokenizer.from_pretrained(Path("tokenizers/bravien-native"))
        assert tok.vocab_size > 0
        assert tok.pad_token_id == 0
        assert tok.bos_token_id == 2
        assert tok.eos_token_id == 3

    # 14. Existing Bravien-v3 rollback still loads
    def test_14_bravien_v3_rollback_loads(self):
        v3_provider = BravienLocalProvider(checkpoint_path="checkpoints/bravien-v3")
        assert v3_provider.checkpoint_path == "checkpoints/bravien-v3"
        meta = v3_provider.metadata()
        assert meta.provider_type == "qwen_compat"
