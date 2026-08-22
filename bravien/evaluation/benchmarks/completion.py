"""Free-form completion scoring (§27).

The model generates greedily and the result is compared against acceptable
answers. Harder than ranking fixed options, and the only one of the two that can
show whether the model produces usable text rather than merely preferring it.

Scoring is string comparison, so the normalisation is the whole design. Two
levels are reported separately and neither is presented as the other:

* **exact** — the normalised completion equals an acceptable answer. Strict, and
  the number to quote.
* **contains** — an acceptable answer appears somewhere in the completion. Looser
  and easy to pass by accident on a long rambling reply, so it is reported
  alongside the mean length that makes it interpretable.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from bravien.evaluation.generation import DETERMINISTIC
from bravien.inference.engine import InferenceEngine
from bravien.utils.logging import get_logger

logger = get_logger("evaluation.completion")

_PUNCTUATION_TAIL = re.compile(r"[\s.,;:!?\"')\]]+$")
_WHITESPACE = re.compile(r"\s+")


def normalise_answer(text: str) -> str:
    """Fold away differences that are not the model's fault.

    Case, surrounding whitespace, repeated internal whitespace, and trailing
    punctuation. Nothing else: stripping articles or stemming would start scoring
    the normaliser's opinions instead of the model's output.
    """
    folded = _WHITESPACE.sub(" ", text.strip()).casefold()
    return _PUNCTUATION_TAIL.sub("", folded)


@dataclass(frozen=True)
class CompletionItem:
    """A prompt and every answer that counts as right."""

    prompt: str
    answers: tuple[str, ...]
    tags: tuple[str, ...] = ()
    max_new_tokens: int = 24

    def __post_init__(self) -> None:
        if not self.answers:
            raise ValueError("a completion item needs at least one accepted answer")
        if not all(a.strip() for a in self.answers):
            raise ValueError("accepted answers must not be blank")


@dataclass
class CompletionOutcome:
    item: CompletionItem
    completion: str
    completion_tokens: int
    finish_reason: str

    @property
    def normalised(self) -> str:
        return normalise_answer(self.completion)

    @property
    def exact(self) -> bool:
        return any(self.normalised == normalise_answer(a) for a in self.item.answers)

    @property
    def contains(self) -> bool:
        return any(normalise_answer(a) in self.normalised for a in self.item.answers)

    def to_dict(self) -> dict[str, object]:
        return {
            "prompt": self.item.prompt,
            "completion": self.completion,
            "accepted": list(self.item.answers),
            "exact": self.exact,
            "contains": self.contains,
            "completion_tokens": self.completion_tokens,
            "finish_reason": self.finish_reason,
            "tags": list(self.item.tags),
        }


@dataclass
class CompletionReport:
    name: str = ""
    outcomes: list[CompletionOutcome] = field(default_factory=list)
    prompt_style: str = "chat"

    def __len__(self) -> int:
        return len(self.outcomes)

    @property
    def exact_matches(self) -> int:
        return sum(1 for o in self.outcomes if o.exact)

    @property
    def exact_rate(self) -> float:
        return self.exact_matches / len(self.outcomes) if self.outcomes else 0.0

    @property
    def contains_rate(self) -> float:
        if not self.outcomes:
            return 0.0
        return sum(1 for o in self.outcomes if o.contains) / len(self.outcomes)

    @property
    def mean_completion_tokens(self) -> float:
        if not self.outcomes:
            return 0.0
        return sum(o.completion_tokens for o in self.outcomes) / len(self.outcomes)

    def to_dict(self) -> dict[str, object]:
        return {
            "suite": self.name,
            "kind": "completion",
            "items": len(self.outcomes),
            "exact": self.exact_matches,
            "exact_rate": round(self.exact_rate, 4),
            "contains_rate": round(self.contains_rate, 4),
            "mean_completion_tokens": round(self.mean_completion_tokens, 2),
            "decoding": "greedy",
            "prompt_style": self.prompt_style,
            "note": "repository-specific probe set, not a standard benchmark",
        }

    def format(self) -> str:
        return (
            f"{self.name or 'completion'}: {self.exact_matches}/{len(self.outcomes)} "
            f"exact = {self.exact_rate:.1%} (contains {self.contains_rate:.1%}, "
            f"mean {self.mean_completion_tokens:.1f} tokens)"
        )


def run_completion(
    engine: InferenceEngine,
    items: Sequence[CompletionItem],
    *,
    name: str = "",
    chat: bool = True,
    system_prompt: str | None = None,
) -> CompletionReport:
    """Generate an answer for each item and score it.

    Args:
        chat: render prompts through the chat template. On by default — a
            completion probe asks a question, and the fine-tuned model was taught
            to answer questions in that format.
    """
    if not items:
        raise ValueError("no items to run")

    report = CompletionReport(name=name, prompt_style="chat" if chat else "raw")

    for item in items:
        if chat:
            messages: list[dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": item.prompt})
            prompt = engine.build_chat_prompt(messages)
        else:
            prompt = item.prompt

        result = engine.complete(
            prompt,
            generation={**DETERMINISTIC, "max_new_tokens": item.max_new_tokens},
        )
        report.outcomes.append(
            CompletionOutcome(
                item=item,
                completion=result.text,
                completion_tokens=result.completion_tokens,
                finish_reason=result.finish_reason,
            )
        )

    logger.info("%s", report.format())
    return report
