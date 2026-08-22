"""The Bravien language model."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from bravien.model.attention import LayerCache, build_causal_mask
from bravien.model.config import BravienConfig
from bravien.model.embeddings import BravienEmbeddings, RotaryEmbedding
from bravien.model.transformer import BravienBlock, build_norm

IGNORE_INDEX = -100


@dataclass
class CausalLMOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    past_key_values: list[LayerCache] | None = None
    hidden_states: torch.Tensor | None = None


@dataclass
class ParameterReport:
    """Where the parameters actually live (§9)."""

    total: int
    trainable: int
    embedding: int
    attention: int
    mlp: int
    norm: int
    lm_head: int

    def as_millions(self, n: int) -> float:
        return n / 1e6


class BravienModel(nn.Module):
    """Embeddings, the transformer stack, and the final norm."""

    def __init__(self, config: BravienConfig) -> None:
        super().__init__()
        self.config = config
        self.embeddings = BravienEmbeddings(config)

        # One rotary module shared by every layer: the tables are a pure
        # function of position, so per-layer copies would be wasted memory.
        self.rotary: RotaryEmbedding | None = None
        if config.position_encoding == "rope":
            self.rotary = RotaryEmbedding(config.head_dim, theta=config.rope_theta)

        self.layers = nn.ModuleList(
            BravienBlock(config, self.rotary) for _ in range(config.num_layers)
        )
        self.final_norm = build_norm(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: list[LayerCache] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, list[LayerCache] | None]:
        bsz, q_len = input_ids.shape
        device = input_ids.device

        past_len = 0
        if past_key_values is not None and len(past_key_values) > 0:
            past_len = past_key_values[0][0].size(2)

        if position_ids is None:
            position_ids = torch.arange(
                past_len, past_len + q_len, device=device, dtype=torch.long
            ).unsqueeze(0).expand(bsz, q_len)

        kv_len = past_len + q_len

        # RoPE extrapolates silently past the trained context: rather than emit
        # quiet nonsense, refuse. Long-context support is a training change, not
        # a forward-pass accident.
        max_ctx = self.config.max_position_embeddings
        if kv_len > max_ctx:
            raise ValueError(
                f"sequence of {kv_len} tokens ({past_len} cached + {q_len} new) "
                f"exceeds max_position_embeddings={max_ctx}"
            )

        # Only materialise a mask when the fast paths in BravienAttention can't
        # be used: padded batches, or appending several tokens onto a cache.
        needs_explicit_mask = attention_mask is not None or (
            q_len > 1 and q_len != kv_len
        )
        attn_mask = (
            build_causal_mask(q_len, kv_len, device, padding_mask=attention_mask)
            if needs_explicit_mask
            else None
        )

        hidden_states = self.embeddings(input_ids, position_ids=position_ids)

        presents: list[LayerCache] = [] if use_cache else None  # type: ignore[assignment]
        for idx, layer in enumerate(self.layers):
            layer_past = past_key_values[idx] if past_key_values is not None else None
            hidden_states, present = layer(
                hidden_states,
                position_ids=position_ids,
                attention_mask=attn_mask,
                past_key_value=layer_past,
                use_cache=use_cache,
            )
            if use_cache:
                presents.append(present)

        return self.final_norm(hidden_states), presents


class BravienForCausalLM(nn.Module):
    """Bravien with a language-modelling head."""

    def __init__(self, config: BravienConfig) -> None:
        super().__init__()
        self.config = config
        self.model = BravienModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            # Share one matrix between input embeddings and output projection.
            # For a small model this is a large fraction of all parameters.
            self.lm_head.weight = self.model.embeddings.token_embeddings.weight

        self.apply(self._init_weights)
        self._scale_residual_projections()

    # ------------------------------------------------------------- init

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def _scale_residual_projections(self) -> None:
        """Damp the projections that write into the residual stream.

        Every layer adds into the same residual path, so without this the
        variance of activations grows with depth. Scaling by 1/sqrt(2L) — two
        residual writes per layer — keeps it roughly constant.
        """
        scale = 1.0 / math.sqrt(2 * self.config.num_layers)
        with torch.no_grad():
            for layer in self.model.layers:
                layer.attention.o_proj.weight.mul_(scale)
                layer.mlp.down_proj.weight.mul_(scale)

    # ---------------------------------------------------------- forward

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: list[LayerCache] | None = None,
        use_cache: bool = False,
    ) -> CausalLMOutput:
        hidden_states, presents = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Causal shift: the logits at position i predict the token at i+1.
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)).float(),
                shift_labels.view(-1),
                ignore_index=IGNORE_INDEX,
            )

        return CausalLMOutput(
            logits=logits,
            loss=loss,
            past_key_values=presents,
            hidden_states=hidden_states,
        )

    # ------------------------------------------------- introspection (§9)

    def parameter_report(self) -> ParameterReport:
        embedding = attention = mlp = norm = lm_head = 0

        emb_ids = set()
        for p in self.model.embeddings.parameters():
            embedding += p.numel()
            emb_ids.add(id(p))

        for layer in self.model.layers:
            for p in layer.attention.parameters():
                attention += p.numel()
            for p in layer.mlp.parameters():
                mlp += p.numel()
            for p in layer.input_norm.parameters():
                norm += p.numel()
            for p in layer.post_attention_norm.parameters():
                norm += p.numel()

        for p in self.model.final_norm.parameters():
            norm += p.numel()

        # A tied head shares storage with the embedding table, so counting it
        # again would inflate the total.
        head_weight = self.lm_head.weight
        if id(head_weight) not in emb_ids:
            lm_head = head_weight.numel()

        total = sum(
            p.numel() for p in {id(p): p for p in self.parameters()}.values()
        )
        trainable = sum(
            p.numel()
            for p in {id(p): p for p in self.parameters()}.values()
            if p.requires_grad
        )

        return ParameterReport(
            total=total,
            trainable=trainable,
            embedding=embedding,
            attention=attention,
            mlp=mlp,
            norm=norm,
            lm_head=lm_head,
        )

    def memory_breakdown(
        self,
        *,
        training: bool,
        batch_size: int = 1,
        seq_len: int | None = None,
        dtype_bytes: int = 2,
        overhead: float = 1.25,
    ) -> dict[str, float]:
        """Estimated VRAM in GB, itemised.

        Returned as a breakdown rather than one number so the figure is
        auditable: if it disagrees with a measured run, you can see which term
        is wrong. Calibrated against measured peaks on an RTX 4050; the
        `overhead` factor covers the SDPA workspace, autograd bookkeeping and
        allocator fragmentation, which are not modelled individually.

        This is an *estimate*. `torch.cuda.max_memory_allocated()` is the
        measurement (§71).
        """
        cfg = self.config
        n = self.parameter_report().total
        seq = seq_len or cfg.max_position_embeddings
        tokens = batch_size * seq
        gb = 1024**3

        if not training:
            kv = self.kv_cache_bytes(
                batch_size=batch_size, seq_len=seq, dtype_bytes=dtype_bytes
            )
            logits = tokens * cfg.vocab_size * dtype_bytes
            parts = {
                "weights": n * dtype_bytes / gb,
                "kv_cache": kv / gb,
                "logits": logits / gb,
            }
            parts["total"] = sum(parts.values())
            return parts

        # fp32 master weights + fp32 grads + two AdamW moments.
        states = n * (4 + 4 + 4 + 4) / gb

        # Tensors each block must keep for its backward pass, per token:
        #   ~6 x hidden (norms, o_proj in/out, residuals, down_proj out)
        #   + q, k, v projections
        #   + 3 x intermediate (gate, up, and their product)
        kv_hidden = cfg.num_kv_heads * cfg.head_dim
        per_token_per_layer = (
            6 * cfg.hidden_size + cfg.hidden_size + 2 * kv_hidden
            + 3 * cfg.intermediate_size
        )
        block_activations = (
            tokens * per_token_per_layer * cfg.num_layers * dtype_bytes / gb
        )

        # Usually the single largest term, and the one most often forgotten:
        # the vocab-wide logits, plus the fp32 copy cross-entropy needs for
        # numerical stability, plus log_softmax's saved output.
        logits = tokens * cfg.vocab_size * dtype_bytes / gb
        loss_workspace = 2 * tokens * cfg.vocab_size * 4 / gb

        parts = {
            "weights_and_optimizer": states,
            "block_activations": block_activations,
            "logits": logits,
            "loss_workspace": loss_workspace,
        }
        subtotal = sum(parts.values())
        parts["overhead"] = subtotal * (overhead - 1.0)
        parts["total"] = subtotal * overhead
        return parts

    def memory_estimate_gb(
        self,
        *,
        training: bool,
        dtype_bytes: int = 2,
        batch_size: int = 1,
        seq_len: int | None = None,
    ) -> float:
        """Estimated footprint in GB. Reported as an estimate, never as measured."""
        return self.memory_breakdown(
            training=training,
            batch_size=batch_size,
            seq_len=seq_len,
            dtype_bytes=dtype_bytes,
        )["total"]

    def kv_cache_bytes(
        self, *, batch_size: int, seq_len: int, dtype_bytes: int = 2
    ) -> int:
        cfg = self.config
        return (
            2  # keys and values
            * cfg.num_layers
            * batch_size
            * cfg.num_kv_heads
            * cfg.head_dim
            * seq_len
            * dtype_bytes
        )

    def describe(self, *, batch_size: int = 1, seq_len: int | None = None) -> str:
        cfg = self.config
        r = self.parameter_report()
        seq = seq_len or cfg.max_position_embeddings

        def count(n: int) -> str:
            """Exact figure first: "0.02 M" tells you nothing about a test model."""
            if n >= 1_000_000:
                return f"{n:,} ({n / 1e6:.2f} M)"
            return f"{n:,}"

        def gb(value: float) -> str:
            return f"{value:.2f} GB" if value >= 0.01 else f"{value * 1024:.1f} MB"

        lines = [
            f"{cfg.name}  (v{cfg.version})",
            "",
            f"  Parameters:      {count(r.total)}",
            f"  Trainable:       {count(r.trainable)}",
            f"    embeddings:    {count(r.embedding)}"
            + ("  (tied to lm_head)" if cfg.tie_word_embeddings else ""),
            f"    attention:     {count(r.attention)}",
            f"    mlp:           {count(r.mlp)}",
            f"    norms:         {count(r.norm)}",
        ]
        if r.lm_head:
            lines.append(f"    lm_head:       {count(r.lm_head)}")
        lines += [
            "",
            f"  Layers:          {cfg.num_layers}",
            f"  Hidden:          {cfg.hidden_size}",
            f"  Heads:           {cfg.num_heads}  (kv: {cfg.num_kv_heads}, "
            f"group: {cfg.num_kv_groups})",
            f"  Head dim:        {cfg.head_dim}",
            f"  Intermediate:    {cfg.intermediate_size}",
            f"  Context:         {cfg.max_position_embeddings}",
            f"  Vocab:           {cfg.vocab_size}",
            f"  Norm / PosEnc:   {cfg.norm_kind} / {cfg.position_encoding}",
            f"  Activation:      {cfg.activation}",
            "",
            f"  Est. train VRAM: {gb(self.memory_estimate_gb(training=True, batch_size=batch_size, seq_len=seq))}"
            f"  (batch {batch_size} x {seq}, bf16, AdamW)",
            f"  Est. infer VRAM: {gb(self.memory_estimate_gb(training=False, batch_size=batch_size, seq_len=seq))}"
            f"  (bf16 weights + full KV cache)",
            "  Estimates, not measurements.",
        ]
        return "\n".join(lines)

    # -------------------------------------------------------- utilities

    @classmethod
    def from_config(cls, config: BravienConfig) -> BravienForCausalLM:
        return cls(config)

    @classmethod
    def from_preset(cls, name: str) -> BravienForCausalLM:
        from bravien.model.config import get_preset

        return cls(get_preset(name))

    def num_parameters(self, trainable_only: bool = False) -> int:
        r = self.parameter_report()
        return r.trainable if trainable_only else r.total
