"""Complete Native Bravien Language Model Architecture and Causal LM Head."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

from bravien.model.bravien_attention import RotaryEmbedding
from bravien.model.bravien_cache import BravienKVCache
from bravien.model.bravien_config import BravienConfig, get_bravien_preset
from bravien.model.bravien_layers import BravienDecoderLayer
from bravien.model.bravien_norm import build_norm

IGNORE_INDEX = -100


@dataclass
class CausalLMOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    past_key_values: BravienKVCache | None = None
    hidden_states: torch.Tensor | None = None


@dataclass
class ParameterReport:
    """Detailed breakdown of where model parameters reside."""

    total: int
    trainable: int
    embedding: int
    attention: int
    mlp: int
    norm: int
    lm_head: int

    @property
    def total_millions(self) -> float:
        return self.total / 1_000_000.0

    @property
    def total_billions(self) -> float:
        return self.total / 1_000_000_000.0

    def summary(self) -> str:
        return (
            f"Bravien Model Parameter Report:\n"
            f"  - Total Parameters:      {self.total:,} ({self.total_millions:.2f}M / {self.total_billions:.3f}B)\n"
            f"  - Trainable Parameters:  {self.trainable:,}\n"
            f"  - Embedding:             {self.embedding:,} ({self.embedding/self.total*100:.1f}%)\n"
            f"  - Attention Layers:      {self.attention:,} ({self.attention/self.total*100:.1f}%)\n"
            f"  - MLP (Feed-Forward):    {self.mlp:,} ({self.mlp/self.total*100:.1f}%)\n"
            f"  - Normalization Layers:  {self.norm:,}\n"
            f"  - Output LM Head:        {self.lm_head:,}"
        )


class BravienEmbeddings(nn.Module):
    """Token embedding layer with optional dropout."""

    def __init__(self, config: BravienConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embed_tokens(input_ids)
        return self.dropout(embeddings)


class BravienModel(nn.Module):
    """Transformer decoder backbone (Embeddings -> Layer Stack -> Final Norm)."""

    def __init__(self, config: BravienConfig) -> None:
        super().__init__()
        self.config = config
        self.embeddings = BravienEmbeddings(config)

        # Global shared rotary embedding instance
        self.rotary: RotaryEmbedding | None = None
        if config.position_encoding == "rope":
            self.rotary = RotaryEmbedding(
                dim=config.head_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_theta,
            )

        self.layers = nn.ModuleList([
            BravienDecoderLayer(config, layer_idx=i) for i in range(config.num_layers)
        ])
        self.final_norm = build_norm(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: BravienKVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, BravienKVCache | None]:
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        past_len = past_key_values.get_seq_length() if past_key_values is not None else 0

        if position_ids is None:
            position_ids = torch.arange(
                past_len, past_len + seq_len, device=device, dtype=torch.long
            ).unsqueeze(0).expand(bsz, seq_len)

        hidden_states = self.embeddings(input_ids)

        for layer in self.layers:
            if self.config.use_gradient_checkpointing and self.training and not use_cache:
                # Custom gradient checkpointing forward
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                hidden_states, _ = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    self.rotary,
                    attention_mask,
                    position_ids,
                    None,
                    False,
                    use_reentrant=False,
                )
            else:
                hidden_states, _ = layer(
                    hidden_states=hidden_states,
                    rotary_emb=self.rotary,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                )

        hidden_states = self.final_norm(hidden_states)
        return hidden_states, past_key_values


class BravienForCausalLM(nn.Module):
    """Bravien Transformer for Causal Language Modeling."""

    def __init__(self, config: BravienConfig) -> None:
        super().__init__()
        self.config = config
        self.model = BravienModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Weight tying if configured (saves ~32k * hidden_size params)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embeddings.embed_tokens.weight

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize parameters with truncated normal distribution (scaled by layer depth)."""
        std = self.config.initializer_range
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                # Scale residual output projections by 1 / sqrt(2 * num_layers) for stability
                if "o_proj" in name or "down_proj" in name:
                    nn.init.normal_(p, mean=0.0, std=std / math.sqrt(2.0 * self.config.num_layers))
                else:
                    nn.init.normal_(p, mean=0.0, std=std)
            elif "weight" in name and ("layernorm" in name or "final_norm" in name):
                nn.init.ones_(p)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embeddings.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.embeddings.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: BravienKVCache | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> CausalLMOutput:
        hidden_states, next_cache = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=IGNORE_INDEX,
            )

        return CausalLMOutput(
            logits=logits,
            loss=loss,
            past_key_values=next_cache,
            hidden_states=hidden_states,
        )

    def count_parameters(self) -> ParameterReport:
        """Count parameters by component."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        embedding = sum(p.numel() for p in self.model.embeddings.parameters())
        attention = sum(
            sum(p.numel() for p in layer.self_attn.parameters()) for layer in self.model.layers
        )
        mlp = sum(
            sum(p.numel() for p in layer.mlp.parameters()) for layer in self.model.layers
        )
        norm = sum(
            sum(p.numel() for p in layer.input_layernorm.parameters())
            + sum(p.numel() for p in layer.post_attention_layernorm.parameters())
            for layer in self.model.layers
        ) + sum(p.numel() for p in self.model.final_norm.parameters())

        lm_head = 0 if self.config.tie_word_embeddings else sum(p.numel() for p in self.lm_head.parameters())

        return ParameterReport(
            total=total,
            trainable=trainable,
            embedding=embedding,
            attention=attention,
            mlp=mlp,
            norm=norm,
            lm_head=lm_head,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Autoregressive generation supporting sampling, greedy decoding, and KV cache."""
        self.eval()
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        eos_token = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        generated = input_ids.clone()
        kv_cache = BravienKVCache.create_empty(self.config.num_layers) if use_cache else None

        # Prefill phase
        if use_cache:
            out = self.forward(input_ids, past_key_values=kv_cache, use_cache=True)
            next_token_logits = out.logits[:, -1, :]
        else:
            out = self.forward(input_ids)
            next_token_logits = out.logits[:, -1, :]

        for _ in range(max_new_tokens):
            logits = next_token_logits / (temperature if temperature > 0.0 else 1.0)

            # Repetition penalty
            if repetition_penalty != 1.0:
                for b in range(bsz):
                    for prev_token in set(generated[b].tolist()):
                        if logits[b, prev_token] < 0:
                            logits[b, prev_token] *= repetition_penalty
                        else:
                            logits[b, prev_token] /= repetition_penalty

            # Greedy or temperature sampling
            if temperature == 0.0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                # Top-K filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float("-inf")

                # Top-P (Nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float("-inf")

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=-1)

            if (next_token == eos_token).all():
                break

            # 1-token decode step
            if use_cache:
                out = self.forward(next_token, past_key_values=kv_cache, use_cache=True)
                next_token_logits = out.logits[:, -1, :]
            else:
                out = self.forward(generated)
                next_token_logits = out.logits[:, -1, :]

        return generated

    def save_pretrained(self, save_directory: str | Path) -> None:
        """Save config and model weights to a standalone directory."""
        from bravien.model.checkpoint import save_bravien_checkpoint
        save_bravien_checkpoint(self, save_directory)

    @classmethod
    def from_pretrained(
        cls,
        save_directory: str | Path,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | str = "auto",
    ) -> BravienForCausalLM:
        """Load native Bravien model from a saved directory."""
        from bravien.model.checkpoint import load_bravien_checkpoint
        return load_bravien_checkpoint(save_directory, device=device, dtype=dtype)
