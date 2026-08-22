"""Multiple choice by log-probability ranking (§27).

The model is never asked to produce a letter. Each option is scored as a
continuation of the same prefix and the highest-scoring one is taken as the
answer. A model that cannot yet write a sentence can still be measured this way,
and the measurement is deterministic — no sampling, no seed, no prompt-format
lottery over whether "(B)" was parsed correctly.

Three things the report insists on, because a benchmark number without them is
easy to overread (§71, §72):

* the **chance rate**, computed from the actual option counts;
* an **exact binomial tail**, so "40% on 20 items" is not mistaken for a result
  when chance is 25%;
* the **normalisation** used, since summed and per-token ranking are different
  measurements and only one of them is quoted.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from bravien.evaluation.scoring import ContinuationScore, rank_options
from bravien.inference.engine import InferenceEngine
from bravien.utils.logging import get_logger

logger = get_logger("evaluation.multiple_choice")


@dataclass(frozen=True)
class MultipleChoiceItem:
    """One question, its options, and the index of the right one."""

    prefix: str
    options: tuple[str, ...]
    answer: int
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.options) < 2:
            raise ValueError("a multiple choice item needs at least two options")
        if not 0 <= self.answer < len(self.options):
            raise ValueError(
                f"answer {self.answer} is outside the {len(self.options)} options"
            )
        if len(set(self.options)) != len(self.options):
            # Duplicated options make the item unscoreable: two identical strings
            # get identical scores, so "correct" would depend on tie-break order.
            raise ValueError("options must be distinct")

    @property
    def chance(self) -> float:
        return 1.0 / len(self.options)


@dataclass
class MultipleChoiceOutcome:
    item: MultipleChoiceItem
    predicted: int
    scores: list[ContinuationScore]

    @property
    def correct(self) -> bool:
        return self.predicted == self.item.answer

    @property
    def margin(self) -> float:
        """How far ahead the chosen option was, per token.

        Near zero means the model effectively had no preference, which is worth
        distinguishing from a confident wrong answer.
        """
        ranked = sorted((s.mean_logprob for s in self.scores), reverse=True)
        return ranked[0] - ranked[1]

    def to_dict(self) -> dict[str, object]:
        return {
            "prefix": self.item.prefix,
            "predicted": self.item.options[self.predicted],
            "expected": self.item.options[self.item.answer],
            "correct": self.correct,
            "margin": round(self.margin, 4),
            "tags": list(self.item.tags),
        }


def binomial_tail(successes: int, trials: int, rate: float) -> float:
    """P(at least `successes` correct | guessing at `rate`).

    Exact rather than a normal approximation: these suites have tens of items, not
    thousands, and the approximation is worst exactly there.
    """
    if trials <= 0:
        return 1.0
    rate = min(max(rate, 0.0), 1.0)
    if rate <= 0.0:
        return 0.0 if successes > 0 else 1.0
    if rate >= 1.0:
        return 1.0

    total = 0.0
    for k in range(successes, trials + 1):
        total += (
            math.comb(trials, k) * (rate**k) * ((1.0 - rate) ** (trials - k))
        )
    return min(total, 1.0)


@dataclass
class MultipleChoiceReport:
    name: str = ""
    outcomes: list[MultipleChoiceOutcome] = field(default_factory=list)
    normalised: bool = True
    prompt_style: str = "raw"

    def __len__(self) -> int:
        return len(self.outcomes)

    @property
    def correct(self) -> int:
        return sum(1 for o in self.outcomes if o.correct)

    @property
    def accuracy(self) -> float:
        return self.correct / len(self.outcomes) if self.outcomes else 0.0

    @property
    def chance(self) -> float:
        """Mean chance rate, which is not 1/n when items have different lengths."""
        if not self.outcomes:
            return 0.0
        return sum(o.item.chance for o in self.outcomes) / len(self.outcomes)

    @property
    def p_value(self) -> float:
        return binomial_tail(self.correct, len(self.outcomes), self.chance)

    @property
    def above_chance(self) -> bool:
        """Whether the result is distinguishable from guessing at p < 0.05.

        False is the honest answer for most of these suites at this model size,
        and reporting it is the point (§71).
        """
        return self.p_value < 0.05

    @property
    def mean_margin(self) -> float:
        if not self.outcomes:
            return 0.0
        return sum(o.margin for o in self.outcomes) / len(self.outcomes)

    def by_tag(self) -> dict[str, float]:
        buckets: dict[str, list[bool]] = {}
        for outcome in self.outcomes:
            for tag in outcome.item.tags:
                buckets.setdefault(tag, []).append(outcome.correct)
        return {
            tag: sum(results) / len(results) for tag, results in sorted(buckets.items())
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "suite": self.name,
            "kind": "multiple_choice",
            "items": len(self.outcomes),
            "correct": self.correct,
            "accuracy": round(self.accuracy, 4),
            "chance": round(self.chance, 4),
            "p_value": round(self.p_value, 5),
            "above_chance": self.above_chance,
            "mean_margin": round(self.mean_margin, 4),
            "ranking": "mean_logprob" if self.normalised else "sum_logprob",
            "prompt_style": self.prompt_style,
            "by_tag": {tag: round(v, 4) for tag, v in self.by_tag().items()},
            "note": "repository-specific probe set, not a standard benchmark",
        }

    def format(self) -> str:
        verdict = "above chance" if self.above_chance else "not distinguishable from chance"
        return (
            f"{self.name or 'multiple choice'}: {self.correct}/{len(self.outcomes)} "
            f"= {self.accuracy:.1%} (chance {self.chance:.1%}, p={self.p_value:.3f}, "
            f"{verdict})"
        )


def run_multiple_choice(
    engine: InferenceEngine,
    items: Sequence[MultipleChoiceItem],
    *,
    name: str = "",
    normalise: bool = True,
    system_prompt: str | None = None,
    chat: bool = False,
) -> MultipleChoiceReport:
    """Score every item and aggregate.

    Args:
        chat: render each prefix through the chat template first. Off by default:
            a raw prefix measures the pretrained language model, which is what
            most of these probes are about. Turn it on to measure the fine-tuned
            model the way a user reaches it.
        system_prompt: only meaningful with `chat`.
    """
    if not items:
        raise ValueError("no items to run")

    report = MultipleChoiceReport(
        name=name, normalised=normalise, prompt_style="chat" if chat else "raw"
    )

    for item in items:
        prefix = item.prefix
        if chat:
            messages: list[dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": item.prefix})
            prefix = engine.build_chat_prompt(messages)

        predicted, scores = rank_options(
            engine, prefix, item.options, normalise=normalise
        )
        report.outcomes.append(
            MultipleChoiceOutcome(item=item, predicted=predicted, scores=scores)
        )

    logger.info("%s", report.format())
    return report
