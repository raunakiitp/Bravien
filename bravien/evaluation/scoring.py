"""Conditional log-probability scoring (§27).

Every objective measurement in this package reduces to one question: how likely
does the model think this text is, given that text? `bravien.inference` answers a
weaker version of it — the mean log-probability of a whole string — which is
enough for perplexity and no use for a benchmark, because a benchmark needs the
probability of an *answer* given a *question*.

Two details make or break the numbers here.

**The off-by-one.** `logits[t]` predicts `input_ids[t + 1]`. A continuation
starting at absolute index `p` is therefore scored from `logits[p - 1 : -1]`, and
getting this wrong produces a score that looks plausible and ranks answers by the
wrong thing entirely.

**Length normalisation.** Summed log-probability always favours the shortest
option, so an unnormalised multiple-choice score measures answer length more than
answer quality. Both are reported; the benchmarks default to the per-token mean
and say so in the report.

**Ties are not answers.** `rank_options` reports every index sharing the top
score instead of resolving to one. Breaking a tie by index silently credits the
tie-break rule to the model, and if a suite happens to keep its correct option at
a fixed position that reads as accuracy (§71).

Scoring never samples, so nothing in this module depends on a seed (§27).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from bravien.inference.engine import InferenceEngine
from bravien.utils.logging import get_logger

logger = get_logger("evaluation.scoring")


class ScoringError(ValueError):
    pass


@dataclass(frozen=True)
class ContinuationScore:
    """How likely `continuation` is, given `prefix`.

    `logprob` is the sum over continuation tokens and `mean_logprob` the
    per-token average. `tokens` is what was actually scored, which is what makes
    the two comparable across a set of options.
    """

    logprob: float
    mean_logprob: float
    tokens: int
    prefix_tokens: int
    truncated_prefix: bool = False

    @property
    def perplexity(self) -> float:
        return float(torch.exp(torch.tensor(-self.mean_logprob)))


@torch.inference_mode()
def score_continuation(
    engine: InferenceEngine, prefix: str, continuation: str
) -> ContinuationScore:
    """Log-probability of `continuation` under the model, given `prefix`.

    The prefix is truncated from the left when the pair does not fit the context
    window: dropping old context changes the conditioning, while truncating the
    continuation would change the question. A continuation that cannot fit on its
    own is an error rather than a silent partial score.
    """
    tokenizer = engine.tokenizer
    continuation_ids = tokenizer.encode(continuation, add_special_tokens=False)
    if not continuation_ids:
        raise ScoringError("continuation encoded to zero tokens")

    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    if not prefix_ids:
        # Position 0 has no predictor, so a bare continuation would lose its
        # first token. BOS is what the model saw in that position during
        # training, so it is the honest stand-in for "nothing came before".
        prefix_ids = [tokenizer.bos_token_id]

    room = engine.max_context - len(continuation_ids)
    if room < 1:
        raise ScoringError(
            f"continuation is {len(continuation_ids)} tokens, which does not fit "
            f"in the {engine.max_context}-token context with anything before it"
        )

    truncated = len(prefix_ids) > room
    if truncated:
        prefix_ids = prefix_ids[-room:]

    ids = torch.tensor(
        [prefix_ids + continuation_ids], dtype=torch.long, device=engine.device
    )
    logits = engine.model(input_ids=ids).logits.to(torch.float32)

    # logits[t] predicts ids[t + 1], so the token at index p is scored by the
    # distribution at p - 1.
    start = len(prefix_ids)
    predictions = logits[0, start - 1 : -1, :]
    targets = ids[0, start:]
    logprobs = torch.log_softmax(predictions, dim=-1).gather(
        -1, targets.unsqueeze(-1)
    ).squeeze(-1)

    total = float(logprobs.sum().item())
    return ContinuationScore(
        logprob=total,
        mean_logprob=total / len(continuation_ids),
        tokens=len(continuation_ids),
        prefix_tokens=len(prefix_ids),
        truncated_prefix=truncated,
    )


def score_options(
    engine: InferenceEngine, prefix: str, options: Sequence[str]
) -> list[ContinuationScore]:
    """Score every option against the same prefix.

    One forward pass per option. Batching them would need padding plus an
    attention mask, and at evaluation sizes the sequential version is fast enough
    that the extra machinery would only add a way to be wrong.
    """
    if not options:
        raise ScoringError("no options to score")
    return [score_continuation(engine, prefix, option) for option in options]


#: Log-probability difference below which two options count as tied.
#:
#: Small enough to mean "the same number, up to float32 accumulation noise"
#: rather than "close": a genuinely close call is a real preference and belongs
#: in the margin, not in the tie count. A uniform model scores every option at
#: exactly -log(vocab_size), so the pathological case lands well inside this.
TIE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class OptionRanking:
    """Which option won, and whether anything actually won.

    `tied` holds every index within `TIE_TOLERANCE` of the top score. When it has
    more than one entry the model expressed no preference, and `best` is only the
    first of them — crediting it as an answer would be scoring the tie-break rule
    rather than the model (§71).
    """

    best: int
    scores: list[ContinuationScore]
    tied: tuple[int, ...]
    normalised: bool = True

    @property
    def decided(self) -> bool:
        """Whether exactly one option came out on top."""
        return len(self.tied) == 1

    @property
    def margin(self) -> float:
        """Gap between the best and second-best score. Zero when tied."""
        key = (
            (lambda s: s.mean_logprob)
            if self.normalised
            else (lambda s: s.logprob)
        )
        ranked = sorted((key(s) for s in self.scores), reverse=True)
        if len(ranked) < 2:
            return 0.0
        return ranked[0] - ranked[1]


def rank_options(
    engine: InferenceEngine,
    prefix: str,
    options: Sequence[str],
    *,
    normalise: bool = True,
) -> OptionRanking:
    """Rank the options, reporting ties rather than silently breaking them.

    `max()` over the scores would return the lowest tied index, which is not a
    neutral default: a suite whose correct option always sits at index 0 would
    then score a tie as a correct answer, and a model with no preference at all
    would post perfect accuracy. `OptionRanking.tied` carries every index sharing
    the top score so the caller can decline to credit an undecided item.

    Args:
        normalise: rank by per-token mean rather than by sum. Default True; a
            summed log-probability ranks by brevity as much as by fit.
    """
    scores = score_options(engine, prefix, options)
    key = (lambda s: s.mean_logprob) if normalise else (lambda s: s.logprob)
    values = [key(score) for score in scores]

    top = max(values)
    tied = tuple(i for i, v in enumerate(values) if top - v <= TIE_TOLERANCE)

    return OptionRanking(
        best=tied[0], scores=scores, tied=tied, normalised=normalise
    )
