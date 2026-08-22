"""Multiple choice by log-probability ranking (§27).

The model is never asked to produce a letter. Each option is scored as a
continuation of the same prefix and the highest-scoring one is taken as the
answer. A model that cannot yet write a sentence can still be measured this way,
and the measurement is deterministic — no sampling, no seed, no prompt-format
lottery over whether "(B)" was parsed correctly.

Four things the report insists on, because a benchmark number without them is
easy to overread (§71, §72):

* the **chance rate**, computed from the actual option counts;
* an **exact binomial tail**, so "40% on 20 items" is not mistaken for a result
  when chance is 25%;
* the **normalisation** used, since summed and per-token ranking are different
  measurements and only one of them is quoted;
* the **position histograms**, and a count of **undecided** items. A tie is not
  an answer and is counted wrong, because breaking it by index would score the
  tie-break rule; and a suite whose answers all sit at one index cannot tell
  accuracy apart from a preference for that position. `balance_answer_positions`
  removes the second problem at the data level.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

from bravien.evaluation.scoring import (
    ContinuationScore,
    OptionRanking,
    rank_options,
)
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
    ranking: OptionRanking

    @property
    def predicted(self) -> int:
        """The top-scoring index. Meaningless on its own when `undecided`."""
        return self.ranking.best

    @property
    def scores(self) -> list[ContinuationScore]:
        return self.ranking.scores

    @property
    def undecided(self) -> bool:
        """Whether the model scored two or more options identically."""
        return not self.ranking.decided

    @property
    def correct(self) -> bool:
        """Right answer, and an actual preference for it.

        An undecided item is never correct. Counting one would credit the
        tie-break rule rather than the model — and because a tie-break lands on
        the lowest index, a suite with its answers at a fixed position would read
        as accuracy (§71).
        """
        return self.ranking.decided and self.predicted == self.item.answer

    @property
    def margin(self) -> float:
        """How far ahead the chosen option was, per token.

        Zero means the model effectively had no preference, which is worth
        distinguishing from a confident wrong answer.
        """
        return self.ranking.margin

    def to_dict(self) -> dict[str, object]:
        return {
            "prefix": self.item.prefix,
            "predicted": self.item.options[self.predicted],
            "expected": self.item.options[self.item.answer],
            "correct": self.correct,
            "undecided": self.undecided,
            "tied_options": len(self.ranking.tied),
            "answer_index": self.item.answer,
            "predicted_index": self.predicted,
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

    @property
    def undecided(self) -> int:
        """Items where two or more options scored identically.

        A non-zero count here caps what the accuracy can mean: those items were
        not answered, and a suite that is mostly ties is measuring nothing (§71).
        """
        return sum(1 for o in self.outcomes if o.undecided)

    def predicted_positions(self) -> dict[int, int]:
        """How often each option index was chosen, ties included.

        The cheapest way to catch a model that always takes the first option, and
        a benchmark whose answers always sit there. Both look like accuracy in the
        headline number and neither survives this histogram.
        """
        counts: dict[int, int] = {}
        for outcome in self.outcomes:
            counts[outcome.predicted] = counts.get(outcome.predicted, 0) + 1
        return dict(sorted(counts.items()))

    def answer_positions(self) -> dict[int, int]:
        """How often the *correct* option sits at each index.

        Everything concentrated at one index means the suite cannot tell accuracy
        apart from position bias, whatever the model does.
        """
        counts: dict[int, int] = {}
        for outcome in self.outcomes:
            index = outcome.item.answer
            counts[index] = counts.get(index, 0) + 1
        return dict(sorted(counts.items()))

    @property
    def answer_positions_balanced(self) -> bool:
        """Whether the correct option appears at more than one index."""
        return len(self.answer_positions()) > 1

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
            "undecided": self.undecided,
            "ranking": "mean_logprob" if self.normalised else "sum_logprob",
            "prompt_style": self.prompt_style,
            "by_tag": {tag: round(v, 4) for tag, v in self.by_tag().items()},
            "predicted_positions": self.predicted_positions(),
            "answer_positions": self.answer_positions(),
            "answer_positions_balanced": self.answer_positions_balanced,
            "note": "repository-specific probe set, not a standard benchmark",
        }

    def format(self) -> str:
        verdict = "above chance" if self.above_chance else "not distinguishable from chance"
        line = (
            f"{self.name or 'multiple choice'}: {self.correct}/{len(self.outcomes)} "
            f"= {self.accuracy:.1%} (chance {self.chance:.1%}, p={self.p_value:.3f}, "
            f"{verdict})"
        )
        # Both caveats change how the accuracy should be read, so they travel with
        # it rather than living further down the report.
        if self.undecided:
            line += f"; {self.undecided} undecided (tied scores, counted wrong)"
        if not self.answer_positions_balanced and len(self.outcomes) > 1:
            line += "; answers all at one index — position bias not measurable"
        return line


def balance_answer_positions(
    items: Sequence[MultipleChoiceItem], *, seed: int
) -> tuple[MultipleChoiceItem, ...]:
    """Permute each item's options so the correct one is not always at one index.

    Written items naturally put the right answer first, which makes the whole
    suite unscoreable: first-position bias, and any tie-break by index, then
    reads as accuracy. Permuting fixes it at the data level, so no scorer has to
    compensate.

    The permutation is derived from the item's own prefix rather than from its
    position in the list, so adding, removing or reordering items leaves every
    other item's layout untouched — a report stays comparable to an older one
    (§27, "evaluation must be reproducible").
    """
    balanced: list[MultipleChoiceItem] = []
    for item in items:
        digest = hashlib.sha256(f"{seed}:{item.prefix}".encode()).digest()
        order = list(range(len(item.options)))
        random.Random(int.from_bytes(digest[:8], "big")).shuffle(order)

        balanced.append(
            replace(
                item,
                options=tuple(item.options[i] for i in order),
                answer=order.index(item.answer),
            )
        )
    return tuple(balanced)


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

        ranking = rank_options(engine, prefix, item.options, normalise=normalise)
        report.outcomes.append(MultipleChoiceOutcome(item=item, ranking=ranking))

    # Both of these make the accuracy mean less than it appears to, so they are
    # said out loud at the point of measurement and not only buried in the JSON.
    if not report.answer_positions_balanced and len(report.outcomes) > 1:
        logger.warning(
            "suite %r has every correct answer at index %d: accuracy here cannot "
            "be told apart from a preference for that position",
            name or "multiple choice",
            next(iter(report.answer_positions())),
        )
    if report.undecided:
        logger.warning(
            "%s: %d of %d items had tied scores and are counted wrong; the model "
            "expressed no preference on them",
            name or "multiple choice",
            report.undecided,
            len(report.outcomes),
        )

    logger.info("%s", report.format())
    return report
