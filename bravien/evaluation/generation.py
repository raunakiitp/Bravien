"""Behavioural measurements of generation (§27, §71).

Perplexity says how well the model predicts text it is shown. It says nothing
about whether the model *stops*, whether it repeats itself, or whether it emits
anything at all — and those are the failures a user notices first. A checkpoint
that improves its eval loss while learning to loop forever has got worse, and
only these metrics show it.

Everything here is counted from real generations. Greedy decoding is the default
so a rerun of the same checkpoint gives the same numbers; sampling would turn
every comparison into noise (§27).

None of these are quality judgements. "Terminated, 14 tokens, no repetition" does
not mean the answer was right — it means the answer was shaped like an answer.
Correctness is what the benchmark suites are for, and neither can be inferred from
the other (§71, §72).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from bravien.inference.engine import InferenceEngine
from bravien.tokenizer.special_tokens import SPECIAL_TOKENS
from bravien.utils.logging import get_logger

logger = get_logger("evaluation.generation")

#: Decoding used for every behavioural measurement. Greedy and unseeded: there is
#: nothing to seed, which is the point.
DETERMINISTIC = {"temperature": 0.0, "do_sample": False, "repetition_penalty": 1.0}


def distinct_ngram_ratio(text: str, n: int = 3) -> float:
    """Fraction of n-grams in `text` that are distinct.

    1.0 means nothing repeated; a low value means the model is looping. Measured
    over whitespace tokens rather than characters, because character n-grams in
    English repeat heavily even in perfectly good prose.

    Text with fewer than `n` words has no n-grams and scores 1.0 — nothing has
    repeated, which is the honest answer for "the" as much as for a full sentence.
    """
    words = text.split()
    if len(words) < n:
        return 1.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return len(set(grams)) / len(grams)


def longest_repeat_run(text: str) -> int:
    """Length of the longest run of one word repeated back to back.

    Catches the specific degenerate mode `distinct_ngram_ratio` is weakest at:
    "the the the the" has few n-grams, and this counts it directly.
    """
    words = text.split()
    if not words:
        return 0
    longest = run = 1
    for previous, current in zip(words, words[1:]):
        run = run + 1 if current == previous else 1
        longest = max(longest, run)
    return longest


def leaks_role_markers(text: str) -> bool:
    """Whether a completion contains a literal role marker.

    Special *ids* cannot appear — the stream decoder drops them. But a model can
    spell `<ASSISTANT>` out of ordinary bytes, and a UI that trusted the text
    would render a forged turn boundary. Worth measuring rather than assuming
    (§52, §53).
    """
    return any(token in text for token in SPECIAL_TOKENS)


@dataclass
class GenerationSample:
    """One prompt, one completion, and what was measured about it."""

    prompt: str
    text: str
    completion_tokens: int
    finish_reason: str
    seconds: float
    first_token_seconds: float = 0.0

    @property
    def terminated(self) -> bool:
        """Whether the model chose to stop rather than hitting the cap."""
        return self.finish_reason in ("stop", "stop_string")

    @property
    def empty(self) -> bool:
        return not self.text.strip()


@dataclass
class BehaviourReport:
    """Aggregates over a set of prompts. Counts and means only."""

    samples: list[GenerationSample] = field(default_factory=list)
    max_new_tokens: int = 0

    def __len__(self) -> int:
        return len(self.samples)

    def _mean(self, values: Sequence[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    @property
    def termination_rate(self) -> float:
        return self._mean([float(s.terminated) for s in self.samples])

    @property
    def empty_rate(self) -> float:
        return self._mean([float(s.empty) for s in self.samples])

    @property
    def mean_completion_tokens(self) -> float:
        return self._mean([float(s.completion_tokens) for s in self.samples])

    @property
    def mean_distinct_trigrams(self) -> float:
        return self._mean([distinct_ngram_ratio(s.text) for s in self.samples])

    @property
    def worst_repeat_run(self) -> int:
        return max((longest_repeat_run(s.text) for s in self.samples), default=0)

    @property
    def marker_leak_rate(self) -> float:
        return self._mean([float(leaks_role_markers(s.text)) for s in self.samples])

    @property
    def tokens_per_second(self) -> float:
        tokens = sum(s.completion_tokens for s in self.samples)
        seconds = sum(s.seconds for s in self.samples)
        return tokens / seconds if seconds > 0 else 0.0

    @property
    def mean_first_token_seconds(self) -> float:
        return self._mean([s.first_token_seconds for s in self.samples])

    def to_dict(self) -> dict[str, object]:
        return {
            "prompts": len(self.samples),
            "max_new_tokens": self.max_new_tokens,
            "termination_rate": round(self.termination_rate, 4),
            "empty_rate": round(self.empty_rate, 4),
            "mean_completion_tokens": round(self.mean_completion_tokens, 2),
            "mean_distinct_trigrams": round(self.mean_distinct_trigrams, 4),
            "worst_repeat_run": self.worst_repeat_run,
            "marker_leak_rate": round(self.marker_leak_rate, 4),
            "tokens_per_second": round(self.tokens_per_second, 2),
            "mean_first_token_seconds": round(self.mean_first_token_seconds, 4),
            "decoding": "greedy",
            # Said plainly so a reader cannot mistake shape for substance (§71).
            "measures": "response shape and cost, not correctness",
        }

    def format(self) -> str:
        return (
            f"{len(self.samples)} prompts: terminated {self.termination_rate:.0%}, "
            f"empty {self.empty_rate:.0%}, mean {self.mean_completion_tokens:.1f} "
            f"tokens, distinct trigrams {self.mean_distinct_trigrams:.2f}, "
            f"{self.tokens_per_second:.1f} tok/s"
        )


def evaluate_behaviour(
    engine: InferenceEngine,
    prompts: Sequence[str],
    *,
    max_new_tokens: int = 64,
    chat: bool = True,
    system_prompt: str | None = None,
) -> BehaviourReport:
    """Generate for each prompt and measure the shape of what came back.

    Args:
        chat: send prompts through the chat template, which is how a user reaches
            the model. Set False to measure raw continuation instead.
        system_prompt: prepended when `chat` is set. Fine-tuning data carries a
            system turn, so omitting it here would measure the model off its own
            training distribution.
    """
    if not prompts:
        raise ValueError("no prompts to evaluate")

    report = BehaviourReport(max_new_tokens=max_new_tokens)
    generation = {**DETERMINISTIC, "max_new_tokens": max_new_tokens}

    for prompt in prompts:
        if chat:
            messages: list[dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            rendered = engine.build_chat_prompt(messages)
        else:
            rendered = prompt

        started = time.perf_counter()
        first_token = 0.0
        text_parts: list[str] = []
        completion_tokens = 0
        finish_reason = "length"

        for event in engine.stream(rendered, generation=dict(generation)):
            if event.done:
                completion_tokens = int(event.usage.get("completion_tokens", 0))
                finish_reason = event.finish_reason
                text_parts = [str(event.usage.get("text", ""))]
            elif first_token == 0.0:
                first_token = time.perf_counter() - started

        report.samples.append(
            GenerationSample(
                prompt=prompt,
                text="".join(text_parts),
                completion_tokens=completion_tokens,
                finish_reason=finish_reason,
                seconds=time.perf_counter() - started,
                first_token_seconds=first_token,
            )
        )

    logger.info("%s", report.format())
    return report
