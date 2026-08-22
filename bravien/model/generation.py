"""Autoregressive generation: sampling primitives and the decode loop."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import torch

from bravien.model.attention import LayerCache
from bravien.model.model import BravienForCausalLM


@dataclass
class GenerationConfig:
    """Decoding parameters (§30)."""

    max_new_tokens: int = 256
    temperature: float = 0.8
    top_k: int | None = 50
    top_p: float | None = 0.95
    repetition_penalty: float = 1.1
    do_sample: bool = True
    #: Ids that end generation. Empty by default because the model layer has no
    #: tokenizer: `bravien.inference.engine` populates this from the tokenizer's
    #: EOS and role-close ids. Leaving it empty means decoding runs to
    #: `max_new_tokens`.
    stop_token_ids: tuple[int, ...] = ()
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be positive or None")
        if self.top_p is not None and not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1] or None")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
        self.stop_token_ids = tuple(self.stop_token_ids)

    @property
    def greedy(self) -> bool:
        """Temperature 0 is treated as a request for greedy decoding."""
        return not self.do_sample or self.temperature == 0.0


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated: Sequence[int] | torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    """Discourage tokens already produced.

    Positive logits are divided and negative logits multiplied, so the penalty
    always pushes a score toward less likely regardless of its sign.
    """
    if penalty == 1.0:
        return logits

    if isinstance(generated, torch.Tensor):
        seen = generated.reshape(-1).to(device=logits.device, dtype=torch.long)
    else:
        seen = torch.tensor(list(generated), device=logits.device, dtype=torch.long)
    if seen.numel() == 0:
        return logits

    unique = torch.unique(seen)
    selected = logits.index_select(-1, unique)
    adjusted = torch.where(selected > 0, selected / penalty, selected * penalty)
    return logits.index_copy(-1, unique, adjusted)


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0 or k >= logits.size(-1):
        return logits
    threshold = torch.topk(logits, k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus filtering: keep the smallest set of tokens with cumulative
    probability >= p."""
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = probs.cumsum(dim=-1)

    # Drop everything strictly past the point where we cross p, but always keep
    # at least the single most likely token.
    remove = cumulative - probs > p
    remove[..., 0] = False

    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    return torch.empty_like(logits).scatter_(-1, sorted_idx, sorted_logits)


def sample_token(
    logits: torch.Tensor,
    config: GenerationConfig,
    generated: Sequence[int],
    generator: torch.Generator | None = None,
) -> int:
    """Turn a (vocab,) logit vector into one token id."""
    logits = logits.to(torch.float32)
    logits = apply_repetition_penalty(logits, generated, config.repetition_penalty)

    if config.greedy:
        return int(torch.argmax(logits, dim=-1).item())

    logits = logits / config.temperature
    if config.top_k is not None:
        logits = top_k_filter(logits, config.top_k)
    if config.top_p is not None:
        logits = top_p_filter(logits, config.top_p)

    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1, generator=generator).item())


@torch.inference_mode()
def generate(
    model: BravienForCausalLM,
    input_ids: torch.Tensor,
    config: GenerationConfig | None = None,
) -> Iterator[int]:
    """Yield generated token ids one at a time.

    A generator rather than a list so the API layer can stream tokens as they
    are produced (§32) instead of waiting for the whole completion.

    `input_ids` is (1, prompt_len) — batch size 1. Batched generation lives in
    `bravien.inference.batching`.
    """
    config = config or GenerationConfig()
    if input_ids.dim() != 2 or input_ids.size(0) != 1:
        raise ValueError(f"expected input_ids shaped (1, T), got {tuple(input_ids.shape)}")

    was_training = model.training
    model.eval()

    generator: torch.Generator | None = None
    if config.seed is not None:
        generator = torch.Generator(device=input_ids.device)
        generator.manual_seed(config.seed)

    device = input_ids.device
    max_ctx = model.config.max_position_embeddings
    prompt_len = input_ids.size(1)
    if prompt_len >= max_ctx:
        raise ValueError(
            f"prompt of {prompt_len} tokens leaves no room in a "
            f"{max_ctx}-token context window"
        )

    stop = set(config.stop_token_ids)
    generated: list[int] = []

    try:
        # Prefill: one pass over the whole prompt, populating the KV cache.
        out = model(input_ids, use_cache=True)
        past: list[LayerCache] | None = out.past_key_values
        next_logits = out.logits[0, -1, :]

        for _ in range(config.max_new_tokens):
            token = sample_token(next_logits, config, generated, generator)
            generated.append(token)
            yield token

            if token in stop:
                return
            if prompt_len + len(generated) >= max_ctx:
                return

            # Decode: feed back only the new token; the cache supplies the rest.
            step_ids = torch.tensor([[token]], device=device, dtype=torch.long)
            out = model(step_ids, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_logits = out.logits[0, -1, :]
    finally:
        if was_training:
            model.train()
