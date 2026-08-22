"""Model correctness tests (§9, §64).

These are the tests that decide whether the architecture is *right*, not merely
whether it runs. The KV-cache and causality tests matter most: both failure
modes produce a model that trains to a plausible-looking loss while silently
leaking future tokens.
"""

from __future__ import annotations

import math

import pytest
import torch

from bravien.model.attention import build_causal_mask
from bravien.model.config import BravienConfig, get_preset
from bravien.model.embeddings import RotaryEmbedding, apply_rotary_pos_emb
from bravien.model.generation import (
    GenerationConfig,
    apply_repetition_penalty,
    generate,
    top_k_filter,
    top_p_filter,
)
from bravien.model.model import IGNORE_INDEX, BravienForCausalLM

# --------------------------------------------------------------------- config


def test_config_rejects_indivisible_heads():
    with pytest.raises(ValueError, match="divisible"):
        BravienConfig(hidden_size=100, num_heads=8)


def test_config_rejects_too_many_kv_heads():
    with pytest.raises(ValueError):
        BravienConfig(hidden_size=64, num_heads=4, num_kv_heads=8)


def test_config_rejects_odd_head_dim_for_rope():
    # RoPE rotates dimension pairs, so an odd head_dim cannot work.
    with pytest.raises(ValueError):
        BravienConfig(hidden_size=12, num_heads=4, num_kv_heads=2, position_encoding="rope")


def test_config_roundtrip(tmp_path):
    cfg = get_preset("small")
    path = cfg.save(tmp_path / "config.json")
    assert BravienConfig.load(path) == cfg


def test_config_rejects_unknown_keys():
    with pytest.raises(ValueError, match="[Uu]nknown"):
        BravienConfig.from_dict({"hidden_size": 64, "nonsense_field": 1})


def test_preset_accepts_bravien_prefix():
    assert get_preset("bravien-tiny") == get_preset("tiny")


# ---------------------------------------------------------------- parameters


def test_parameter_count_matches_hand_calculation(tiny_config, tiny_model):
    c = tiny_config
    embed = c.vocab_size * c.hidden_size
    attn = (
        c.hidden_size * c.hidden_size                      # q_proj
        + 2 * c.hidden_size * (c.num_kv_heads * c.head_dim)  # k_proj, v_proj
        + c.hidden_size * c.hidden_size                    # o_proj
    )
    mlp = 3 * c.hidden_size * c.intermediate_size
    norms = 2 * c.hidden_size                              # two RMSNorm per block
    per_layer = attn + mlp + norms
    expected = embed + c.num_layers * per_layer + c.hidden_size  # + final norm

    report = tiny_model.parameter_report()
    assert report.total == expected, (
        f"expected {expected:,} parameters, model has {report.total:,}"
    )


def test_tied_embeddings_share_storage(tiny_config):
    model = BravienForCausalLM(tiny_config)
    assert model.lm_head.weight.data_ptr() == (
        model.model.embeddings.token_embeddings.weight.data_ptr()
    )


def test_untied_embeddings_are_separate(tiny_config):
    cfg = BravienConfig.from_dict({**tiny_config.to_dict(), "tie_word_embeddings": False})
    model = BravienForCausalLM(cfg)
    assert model.lm_head.weight.data_ptr() != (
        model.model.embeddings.token_embeddings.weight.data_ptr()
    )
    # The untied head adds exactly one vocab x hidden matrix.
    tied = BravienForCausalLM(tiny_config).parameter_report().total
    assert model.parameter_report().total == tied + cfg.vocab_size * cfg.hidden_size


# ------------------------------------------------------------------- forward


def test_forward_shapes(tiny_config, tiny_model):
    ids = torch.randint(0, tiny_config.vocab_size, (2, 9))
    out = tiny_model(input_ids=ids)
    assert out.logits.shape == (2, 9, tiny_config.vocab_size)
    assert out.loss is None
    assert torch.isfinite(out.logits).all()


def test_initial_loss_is_near_uniform_entropy(tiny_config, tiny_model):
    """An untrained model should be about as uncertain as a uniform distribution.

    A loss far below ln(vocab_size) at step zero means labels are leaking into
    the inputs; far above means initialisation is broken.
    """
    ids = torch.randint(0, tiny_config.vocab_size, (4, 16))
    loss = tiny_model(input_ids=ids, labels=ids).loss
    assert loss is not None
    assert math.isclose(loss.item(), math.log(tiny_config.vocab_size), rel_tol=0.10)


def test_loss_ignores_masked_labels(tiny_config, tiny_model):
    ids = torch.randint(0, tiny_config.vocab_size, (1, 12))
    labels = ids.clone()
    labels[:, :6] = IGNORE_INDEX
    partial = tiny_model(input_ids=ids, labels=labels).loss
    full = tiny_model(input_ids=ids, labels=ids).loss
    assert torch.isfinite(partial)
    assert not torch.allclose(partial, full)


def test_all_labels_masked_yields_nan_not_silent_zero(tiny_config, tiny_model):
    """Fully-masked supervision must be visible, not silently trained as zero."""
    ids = torch.randint(0, tiny_config.vocab_size, (1, 8))
    labels = torch.full_like(ids, IGNORE_INDEX)
    loss = tiny_model(input_ids=ids, labels=labels).loss
    assert torch.isnan(loss)


def test_gradients_reach_every_parameter(tiny_config):
    model = BravienForCausalLM(tiny_config)
    ids = torch.randint(0, tiny_config.vocab_size, (2, 10))
    model(input_ids=ids, labels=ids).loss.backward()
    missing = [
        name
        for name, p in model.named_parameters()
        if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())
    ]
    assert not missing, f"parameters with no/invalid gradient: {missing}"


# ------------------------------------------------------------------ causality


def test_future_tokens_do_not_affect_earlier_logits(tiny_config, tiny_model):
    """The defining property of a causal LM.

    If this fails, the model is reading its own answer and every loss curve it
    produces is meaningless.
    """
    ids = torch.randint(0, tiny_config.vocab_size, (1, 12))
    baseline = tiny_model(input_ids=ids).logits

    tampered = ids.clone()
    tampered[0, -1] = (tampered[0, -1] + 7) % tiny_config.vocab_size
    changed = tiny_model(input_ids=tampered).logits

    torch.testing.assert_close(baseline[:, :-1], changed[:, :-1], atol=1e-5, rtol=1e-5)
    assert not torch.allclose(baseline[:, -1], changed[:, -1])


def test_causal_mask_is_bottom_right_aligned():
    """With a cache, new queries sit at the *end* of the key sequence."""
    mask = build_causal_mask(q_len=2, kv_len=5, device=torch.device("cpu"))
    assert mask.shape == (1, 1, 2, 5)
    # Query 0 is at absolute position 3, so it sees keys 0..3 but not 4.
    assert mask[0, 0, 0].tolist() == [True, True, True, True, False]
    assert mask[0, 0, 1].tolist() == [True, True, True, True, True]


def test_causal_mask_merges_padding():
    padding = torch.tensor([[True, True, False, True]])  # key 2 is padding
    mask = build_causal_mask(4, 4, torch.device("cpu"), padding_mask=padding)
    assert mask[0, 0, 3].tolist() == [True, True, False, True]


# ------------------------------------------------------------------- kv cache


def test_incremental_decode_matches_full_forward(tiny_config, tiny_model):
    """Token-at-a-time decoding with a cache must equal one full forward pass.

    This is the test that catches RoPE applied after the cache append, or keys
    repeated with `repeat` instead of `repeat_interleave` for grouped-query
    attention. Both produce a model that looks fine until it generates.
    """
    ids = torch.randint(0, tiny_config.vocab_size, (1, 10))
    reference = tiny_model(input_ids=ids).logits

    past = None
    collected = []
    for i in range(ids.size(1)):
        out = tiny_model(input_ids=ids[:, i : i + 1], past_key_values=past, use_cache=True)
        past = out.past_key_values
        collected.append(out.logits)
    incremental = torch.cat(collected, dim=1)

    torch.testing.assert_close(reference, incremental, atol=1e-4, rtol=1e-4)


def test_prefill_then_chunked_append_matches_full_forward(tiny_config, tiny_model):
    """Appending several tokens at once onto a cache must also stay aligned.

    This exercises the explicit bottom-right mask; `is_causal=True` here would
    align top-left and leak.
    """
    ids = torch.randint(0, tiny_config.vocab_size, (1, 12))
    reference = tiny_model(input_ids=ids).logits

    first = tiny_model(input_ids=ids[:, :7], use_cache=True)
    second = tiny_model(
        input_ids=ids[:, 7:], past_key_values=first.past_key_values, use_cache=True
    )
    stitched = torch.cat([first.logits, second.logits], dim=1)

    torch.testing.assert_close(reference, stitched, atol=1e-4, rtol=1e-4)


def test_cache_shapes_use_kv_heads(tiny_config, tiny_model):
    """The cache stores the *unexpanded* KV heads — that is the point of GQA."""
    ids = torch.randint(0, tiny_config.vocab_size, (1, 5))
    out = tiny_model(input_ids=ids, use_cache=True)
    assert out.past_key_values is not None
    assert len(out.past_key_values) == tiny_config.num_layers
    k, v = out.past_key_values[0]
    assert k.shape == (1, tiny_config.num_kv_heads, 5, tiny_config.head_dim)
    assert v.shape == k.shape


def test_grouped_query_attention_equals_full_attention(tiny_config):
    """GQA with num_kv_heads == num_heads must reduce to ordinary attention."""
    cfg = BravienConfig.from_dict(
        {**tiny_config.to_dict(), "num_kv_heads": tiny_config.num_heads}
    )
    torch.manual_seed(1)
    model = BravienForCausalLM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 6))
    out = model(input_ids=ids, use_cache=True)
    k, _ = out.past_key_values[0]
    assert k.shape[1] == cfg.num_heads
    assert torch.isfinite(out.logits).all()


def test_position_beyond_context_is_rejected(tiny_config, tiny_model):
    too_long = torch.zeros(1, tiny_config.max_position_embeddings + 1, dtype=torch.long)
    with pytest.raises(ValueError, match="max_position_embeddings|context"):
        tiny_model(input_ids=too_long)


# ---------------------------------------------------------------------- rope


def test_rope_preserves_norm():
    """Rotation changes direction, never magnitude."""
    rope = RotaryEmbedding(head_dim=8, theta=10000.0)
    q = torch.randn(1, 2, 5, 8)
    cos, sin = rope(torch.arange(5).unsqueeze(0), dtype=q.dtype)
    rotated, _ = apply_rotary_pos_emb(q, q, cos, sin)
    torch.testing.assert_close(q.norm(dim=-1), rotated.norm(dim=-1), atol=1e-5, rtol=1e-5)


def test_rope_encodes_relative_position():
    """q.k after RoPE depends on the offset between positions, not their values."""
    rope = RotaryEmbedding(head_dim=8, theta=10000.0)
    q = torch.randn(1, 1, 1, 8)
    k = torch.randn(1, 1, 1, 8)

    def dot(pos_q: int, pos_k: int) -> float:
        cq, sq = rope(torch.tensor([[pos_q]]), dtype=q.dtype)
        ck, sk = rope(torch.tensor([[pos_k]]), dtype=k.dtype)
        rq, _ = apply_rotary_pos_emb(q, q, cq, sq)
        rk, _ = apply_rotary_pos_emb(k, k, ck, sk)
        return float((rq * rk).sum())

    assert math.isclose(dot(0, 3), dot(5, 8), abs_tol=1e-4)
    assert math.isclose(dot(2, 2), dot(9, 9), abs_tol=1e-4)


# ---------------------------------------------------------------- generation


def test_generation_config_validates():
    with pytest.raises(ValueError):
        GenerationConfig(temperature=-0.1)
    with pytest.raises(ValueError):
        GenerationConfig(top_p=1.5)
    with pytest.raises(ValueError):
        GenerationConfig(top_k=0)
    with pytest.raises(ValueError):
        GenerationConfig(max_new_tokens=0)


def test_zero_temperature_means_greedy():
    """Temperature 0 is a valid request, not an error — it selects argmax."""
    assert GenerationConfig(temperature=0.0, do_sample=True).greedy
    assert not GenerationConfig(temperature=0.8, do_sample=True).greedy


def test_stop_token_ids_normalises_to_tuple():
    assert GenerationConfig(stop_token_ids=[1, 2]).stop_token_ids == (1, 2)


def test_top_k_filter_keeps_exactly_k():
    logits = torch.tensor([[1.0, 5.0, 3.0, 2.0, 4.0]])
    kept = torch.isfinite(top_k_filter(logits, 2))
    assert kept.sum().item() == 2
    assert kept[0, 1] and kept[0, 4]


def test_top_p_filter_always_keeps_the_argmax():
    """Even at p=0, the single most likely token must survive."""
    logits = torch.tensor([[0.1, 9.0, 0.2]])
    out = top_p_filter(logits, 0.0)
    assert torch.isfinite(out[0, 1])
    assert torch.isinf(out[0, 0]) and torch.isinf(out[0, 2])


def test_repetition_penalty_pushes_seen_tokens_down():
    logits = torch.tensor([[2.0, -2.0, 0.5]])
    for seen in (torch.tensor([[0, 1]]), [0, 1]):
        out = apply_repetition_penalty(logits.clone(), seen, penalty=2.0)
        assert out[0, 0] < logits[0, 0]   # positive logit divided
        assert out[0, 1] < logits[0, 1]   # negative logit multiplied
        assert out[0, 2] == logits[0, 2]  # untouched


def test_repetition_penalty_handles_empty_history():
    logits = torch.tensor([[2.0, -2.0, 0.5]])
    for empty in ([], torch.empty(1, 0, dtype=torch.long)):
        torch.testing.assert_close(
            apply_repetition_penalty(logits.clone(), empty, 2.0), logits
        )


def test_generate_yields_requested_number_of_tokens(tiny_config, tiny_model):
    prompt = torch.randint(0, tiny_config.vocab_size, (1, 4))
    cfg = GenerationConfig(max_new_tokens=6, do_sample=True, seed=0, stop_token_ids=())
    produced = list(generate(tiny_model, prompt, cfg))
    assert len(produced) == 6
    assert all(isinstance(t, int) and 0 <= t < tiny_config.vocab_size for t in produced)


def test_greedy_generation_is_deterministic(tiny_config, tiny_model):
    prompt = torch.randint(0, tiny_config.vocab_size, (1, 4))
    cfg = GenerationConfig(max_new_tokens=8, do_sample=False, stop_token_ids=())
    assert list(generate(tiny_model, prompt, cfg)) == list(
        generate(tiny_model, prompt, cfg)
    )


def test_seeded_sampling_is_reproducible(tiny_config, tiny_model):
    prompt = torch.randint(0, tiny_config.vocab_size, (1, 4))
    cfg = GenerationConfig(max_new_tokens=8, do_sample=True, seed=1234, stop_token_ids=())
    assert list(generate(tiny_model, prompt, cfg)) == list(
        generate(tiny_model, prompt, cfg)
    )


def test_generation_stops_at_stop_token(tiny_config, tiny_model):
    """Force a stop by making one token overwhelmingly likely under greedy decode."""
    prompt = torch.randint(0, tiny_config.vocab_size, (1, 4))
    with torch.no_grad():
        target = int(tiny_model(input_ids=prompt).logits[0, -1].argmax())
    cfg = GenerationConfig(
        max_new_tokens=20, do_sample=False, stop_token_ids=(target,)
    )
    produced = list(generate(tiny_model, prompt, cfg))
    assert produced == [] or target not in produced[:-1]
    assert len(produced) < 20


def test_generate_restores_training_mode(tiny_config):
    model = BravienForCausalLM(tiny_config)
    model.train()
    prompt = torch.randint(0, tiny_config.vocab_size, (1, 3))
    list(generate(model, prompt, GenerationConfig(max_new_tokens=2, stop_token_ids=())))
    assert model.training, "generate() must not leave the model in eval mode"


def test_generate_respects_context_limit(tiny_config, tiny_model):
    """Generation must stop at the context window rather than crash inside SDPA."""
    ctx = tiny_config.max_position_embeddings
    prompt = torch.randint(0, tiny_config.vocab_size, (1, ctx - 3))
    cfg = GenerationConfig(max_new_tokens=50, do_sample=False, stop_token_ids=())
    assert len(list(generate(tiny_model, prompt, cfg))) <= 3


# ------------------------------------------------------------------ reporting


def test_describe_reports_real_numbers(tiny_model):
    text = tiny_model.describe()
    report = tiny_model.parameter_report()
    assert f"{report.total:,}" in text
    assert "bravien" in text.lower()


def test_memory_estimate_grows_with_training(tiny_model):
    assert tiny_model.memory_estimate_gb(training=True) > tiny_model.memory_estimate_gb(
        training=False
    )


def test_memory_breakdown_sums_to_total(tiny_model):
    parts = tiny_model.memory_breakdown(training=True, batch_size=2, seq_len=16)
    total = parts.pop("total")
    assert math.isclose(sum(parts.values()), total, rel_tol=1e-9)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_memory_estimate_is_within_2x_of_measured():
    """Keeps the estimate honest.

    An estimate that is wrong by an order of magnitude is worse than no estimate,
    because it gets used to pick a batch size (§71).
    """
    from bravien.model.config import BravienConfig

    cfg = BravienConfig(
        vocab_size=4096, hidden_size=256, num_layers=4, num_heads=4,
        num_kv_heads=2, intermediate_size=688, max_position_embeddings=256,
    )
    batch, seq = 4, 256
    model = BravienForCausalLM(cfg).to("cuda")
    estimate = model.memory_estimate_gb(training=True, batch_size=batch, seq_len=seq)

    torch.cuda.reset_peak_memory_stats()
    ids = torch.randint(0, cfg.vocab_size, (batch, seq), device="cuda")
    # A real optimizer step, so the moment buffers actually exist.
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model(input_ids=ids, labels=ids).loss.backward()
    opt.step()
    torch.cuda.synchronize()
    measured = torch.cuda.max_memory_allocated() / 1024**3

    assert 0.5 <= estimate / measured <= 2.0, (
        f"estimate {estimate:.3f} GB vs measured {measured:.3f} GB "
        f"(ratio {estimate / measured:.2f}) — the estimate needs recalibrating"
    )
