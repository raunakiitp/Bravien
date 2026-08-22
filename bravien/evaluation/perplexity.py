"""Perplexity on held-out text (§27).

Perplexity is the one number that tracks pretraining progress honestly, and the
one most often quoted misleadingly. Two rules this module enforces rather than
hopes for:

* **A perplexity is a property of a model *and* a tokenizer.** A smaller vocabulary
  splits text into more tokens, each easier to predict, so the same model can post
  a better number simply by changing how it counts. Every report here carries the
  tokenizer's vocabulary checksum, so two numbers that are not comparable cannot
  be quietly compared (§23, §71).
* **Long documents are scored with a sliding window.** Truncating to the context
  length would report the perplexity of the first N tokens and call it the
  document's. The window advances by a stride and only the newly exposed tokens
  are counted, so every token is scored with as much context as the model has.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import torch

from bravien.inference.engine import InferenceEngine
from bravien.utils.logging import get_logger

logger = get_logger("evaluation.perplexity")


@dataclass
class PerplexityResult:
    """A measurement, plus enough context to know what it may be compared with."""

    loss: float
    perplexity: float
    tokens: int
    documents: int
    skipped: int = 0
    window: int = 0
    stride: int = 0
    tokenizer_checksum: str = ""
    vocab_size: int = 0

    @property
    def bits_per_token(self) -> float:
        """Cross-entropy in bits, which is tokenizer-independent in spirit."""
        return self.loss / math.log(2)

    def to_dict(self) -> dict[str, object]:
        return {
            "loss": round(self.loss, 6),
            "perplexity": round(self.perplexity, 4),
            "bits_per_token": round(self.bits_per_token, 4),
            "tokens": self.tokens,
            "documents": self.documents,
            "skipped": self.skipped,
            "window": self.window,
            "stride": self.stride,
            "tokenizer_checksum": self.tokenizer_checksum,
            "vocab_size": self.vocab_size,
            # Stated rather than implied: a perplexity against a different
            # tokenizer is a different measurement (§71).
            "comparable_only_with_same_tokenizer": True,
        }

    def format(self) -> str:
        return (
            f"perplexity {self.perplexity:.2f} (loss {self.loss:.4f}, "
            f"{self.bits_per_token:.3f} bits/token) over {self.tokens:,} tokens "
            f"in {self.documents:,} documents"
        )


@dataclass
class _Accumulator:
    """Token-weighted mean loss.

    Weighted by token count rather than by document, so a one-line document does
    not carry the same weight as a long one.
    """

    loss_sum: float = 0.0
    tokens: int = 0
    documents: int = 0
    skipped: int = 0
    lengths: list[int] = field(default_factory=list)

    def add(self, loss_sum: float, tokens: int) -> None:
        self.loss_sum += loss_sum
        self.tokens += tokens
        self.documents += 1
        self.lengths.append(tokens)

    @property
    def mean_loss(self) -> float:
        return self.loss_sum / self.tokens if self.tokens else float("nan")


@torch.inference_mode()
def _score_window(
    engine: InferenceEngine, ids: Sequence[int], *, count_from: int
) -> tuple[float, int]:
    """Summed loss over `ids[count_from:]`, conditioned on everything before it.

    Returns (summed negative log-likelihood, tokens counted). Tokens before
    `count_from` are context: they are fed to the model but not scored, which is
    what stops a sliding window from counting the same token twice.
    """
    tensor = torch.tensor([list(ids)], dtype=torch.long, device=engine.device)
    logits = engine.model(input_ids=tensor).logits.to(torch.float32)

    # logits[t] predicts ids[t + 1].
    start = max(count_from, 1)
    predictions = logits[0, start - 1 : -1, :]
    targets = tensor[0, start:]
    if targets.numel() == 0:
        return 0.0, 0

    nll = torch.nn.functional.cross_entropy(
        predictions, targets, reduction="sum"
    )
    return float(nll.item()), int(targets.numel())


def evaluate_perplexity(
    engine: InferenceEngine,
    texts: Iterable[str],
    *,
    stride: int | None = None,
    max_documents: int | None = None,
) -> PerplexityResult:
    """Perplexity over a corpus, one document at a time.

    Args:
        stride: how far the window advances on a document longer than the context.
            Defaults to half the context: every token then gets at least half a
            window of context, at twice the compute of a non-overlapping pass.
        max_documents: stop early. Useful mid-training, where the point is a trend
            rather than a final number.

    Documents shorter than two tokens are skipped and counted: a single token has
    no predecessor to be predicted from.
    """
    window = engine.max_context
    step = stride or max(window // 2, 1)
    if step > window:
        raise ValueError(f"stride {step} exceeds the {window}-token context")

    accumulator = _Accumulator()
    was_training = engine.model.training
    engine.model.eval()

    try:
        for index, text in enumerate(texts):
            if max_documents is not None and index >= max_documents:
                break

            ids = engine.tokenizer.encode(text, add_special_tokens=False)
            if len(ids) < 2:
                accumulator.skipped += 1
                continue

            if len(ids) <= window:
                loss_sum, counted = _score_window(engine, ids, count_from=1)
                accumulator.add(loss_sum, counted)
                continue

            # Sliding window. The first window scores everything after its first
            # token; later windows score only what the previous one could not.
            document_loss = 0.0
            document_tokens = 0
            start = 0
            previous_end = 0
            while start < len(ids):
                end = min(start + window, len(ids))
                chunk = ids[start:end]
                count_from = 1 if start == 0 else previous_end - start
                loss_sum, counted = _score_window(
                    engine, chunk, count_from=count_from
                )
                document_loss += loss_sum
                document_tokens += counted
                previous_end = end
                if end == len(ids):
                    break
                start += step
            accumulator.add(document_loss, document_tokens)
    finally:
        if was_training:
            engine.model.train()

    if accumulator.tokens == 0:
        raise ValueError(
            "nothing was scored: every document was shorter than two tokens"
        )

    mean_loss = accumulator.mean_loss
    try:
        perplexity = math.exp(mean_loss)
    except OverflowError:
        perplexity = float("inf")

    result = PerplexityResult(
        loss=mean_loss,
        perplexity=perplexity,
        tokens=accumulator.tokens,
        documents=accumulator.documents,
        skipped=accumulator.skipped,
        window=window,
        stride=step,
        tokenizer_checksum=str(
            engine.tokenizer.metadata.get("vocab_checksum", "unknown")
        ),
        vocab_size=engine.tokenizer.vocab_size,
    )
    logger.info("%s", result.format())
    return result


def random_baseline_perplexity(vocab_size: int) -> float:
    """What a model that has learned nothing scores.

    Quoted next to every perplexity in the reports: a number is only meaningful
    against the alternative of guessing uniformly (§71).
    """
    return float(vocab_size)
