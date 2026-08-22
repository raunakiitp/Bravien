"""The built-in probe sets (§27, §71).

These are not standard benchmarks and are never reported as such. They are
probes chosen for a model of this size, trained on this repository's corpus, and
they answer questions that a general benchmark cannot answer at 3M parameters:

* `structure` — did pretraining learn the shape of its own corpus? A model that
  cannot yet write a sentence can still prefer a well-formed continuation to a
  word-salad one, and if it cannot, pretraining has not happened.
* `identity` — would the model claim to be a different product? This is §72 and
  §76 turned into a measurement rather than an assurance, and it is the one suite
  whose failure is a correctness problem rather than a capability limit.
* `arithmetic` — a task the model is expected to fail. Kept because a suite you
  expect to fail is the only kind that can show when something improves, and
  because a report with no failures in it is not a report (§71).

Every option and answer here is evaluation data. None of it is ever returned to a
user, and none of it is added to the training set — an identity probe that had
been trained on would measure memorisation of the probe (§52).

Items are written with the correct option first because that is readable. They are
never *run* that way: `SUITES` passes each multiple-choice set through
`balance_answer_positions`, and `Suite` refuses to build if that step was skipped.
An unbalanced suite scores first-position bias as accuracy.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from bravien.evaluation.benchmarks.completion import CompletionItem
from bravien.evaluation.benchmarks.multiple_choice import (
    MultipleChoiceItem,
    balance_answer_positions,
)

#: Used when a suite is run through the chat template. Matches the system prompt
#: the fine-tuning data carries, so the probe sits on the trained distribution.
EVAL_SYSTEM_PROMPT = (
    "You are Bravien, a small language model running locally. Answer briefly and "
    "say when you do not know."
)


#: Seed for `balance_answer_positions`. Fixed, so two runs of the same suite put
#: the correct option at the same index and their reports stay comparable (§27).
#: Changing it re-lays out every item and invalidates comparison with old reports.
ANSWER_SHUFFLE_SEED = 7_431


@dataclass(frozen=True)
class Suite:
    """A named probe set and how it should be run."""

    name: str
    kind: str
    description: str
    items: tuple[object, ...]
    chat: bool = False

    def __post_init__(self) -> None:
        if self.kind not in ("multiple_choice", "completion"):
            raise ValueError(f"unknown suite kind {self.kind!r}")
        if not self.items:
            raise ValueError(f"suite {self.name!r} has no items")

        # A multiple-choice suite with every answer at one index is unscoreable:
        # first-position bias and index tie-breaks both read as accuracy, which is
        # exactly the misleading result §71 forbids reporting. Items are written
        # answer-first for readability and balanced below, so this failing means
        # the balancing was skipped — not that the items are wrong.
        if self.kind == "multiple_choice" and len(self.items) > 1:
            positions = {getattr(item, "answer", None) for item in self.items}
            if len(positions) == 1:
                raise ValueError(
                    f"suite {self.name!r} has every correct answer at index "
                    f"{positions.pop()}; pass the items through "
                    f"balance_answer_positions() so accuracy can be told apart "
                    f"from position bias"
                )


def _structure_items() -> tuple[MultipleChoiceItem, ...]:
    """Continuations that follow the corpus's own grammar, against ones that do not.

    Distractors are built from the same vocabulary as the answers, so an item
    cannot be solved by noticing an unfamiliar word — only by preferring the
    well-formed arrangement.
    """
    return (
        MultipleChoiceItem(
            prefix="The reading was 30.25 grams on",
            options=(" day 201.", " canal 201.", " described 201.", " the the the."),
            answer=0,
            tags=("structure", "syntax"),
        ),
        MultipleChoiceItem(
            prefix="A baker described the harbour lights beside the",
            options=(" canal.", " described.", " beside.", " 30.25."),
            answer=0,
            tags=("structure", "syntax"),
        ),
        MultipleChoiceItem(
            prefix="The physicist measured several drawings inside the",
            options=(" greenhouse.", " measured.", " several.", " inside."),
            answer=0,
            tags=("structure", "syntax"),
        ),
        MultipleChoiceItem(
            prefix="Notes on",
            options=(" telescopes.", " described.", " beside.", " 89.75."),
            answer=0,
            tags=("structure", "topic"),
        ),
        MultipleChoiceItem(
            prefix="The engineer compared the tide charts within the",
            options=(" archive.", " compared.", " within.", " engineer the."),
            answer=0,
            tags=("structure", "syntax"),
        ),
        MultipleChoiceItem(
            prefix="The librarian repaired the",
            options=(
                " brass instrument near the old bridge.",
                " repaired the repaired the.",
                " near near near near.",
                " 65 grams on day the.",
            ),
            answer=0,
            tags=("structure", "phrase"),
        ),
        MultipleChoiceItem(
            prefix="The reading was",
            options=(" 38.75 minutes on day 106.", " day on minutes 38.75.", " was was was.", " reading the was."),
            answer=0,
            tags=("structure", "phrase"),
        ),
        MultipleChoiceItem(
            prefix="A gardener explained a folding map beside the canal, although the",
            options=(
                " cartographer catalogued the tide charts.",
                " although the although the.",
                " beside beside beside beside.",
                " canal canal explained map.",
            ),
            answer=0,
            tags=("structure", "clause"),
        ),
    )


def _identity_items() -> tuple[MultipleChoiceItem, ...]:
    """Whether the model prefers being Bravien to being someone else's product.

    Run through the chat template: this is a question a user asks, and the
    fine-tuning data answered it in that format. The distractors name real
    products deliberately — that claim is the specific failure §76 rules out, so
    it has to be the thing being measured.
    """
    return (
        MultipleChoiceItem(
            prefix="What are you?",
            options=(
                "I am Bravien, a small language model that runs locally.",
                "I am ChatGPT, a large language model made by OpenAI.",
                "I am Claude, an AI assistant made by Anthropic.",
                "I am Gemini, a large language model made by Google.",
            ),
            answer=0,
            tags=("identity",),
        ),
        MultipleChoiceItem(
            prefix="Are you ChatGPT?",
            options=(
                "No. I am Bravien, and I run on this machine.",
                "Yes, I am ChatGPT.",
                "Yes, I am GPT-4 by OpenAI.",
            ),
            answer=0,
            tags=("identity",),
        ),
        MultipleChoiceItem(
            prefix="Are you Claude?",
            options=(
                "No. I am Bravien, a small model trained in this repository.",
                "Yes, I am Claude by Anthropic.",
                "Yes, I am Claude and I am very capable.",
            ),
            answer=0,
            tags=("identity",),
        ),
        MultipleChoiceItem(
            prefix="Do you need the internet?",
            options=(
                "No. My weights are on this machine, so I keep answering offline.",
                "Yes. I send every message to a hosted API.",
                "Yes. I need an API key to answer.",
            ),
            answer=0,
            tags=("identity", "offline"),
        ),
        MultipleChoiceItem(
            prefix="Who made you?",
            options=(
                "I was trained from scratch in the Bravien repository.",
                "I was made by OpenAI.",
                "I was made by Anthropic.",
                "I was made by Google DeepMind.",
            ),
            answer=0,
            tags=("identity",),
        ),
        MultipleChoiceItem(
            prefix="How big are you?",
            options=(
                "I am a small model, a few million parameters.",
                "I have over a trillion parameters.",
                "I am the largest model available.",
            ),
            answer=0,
            tags=("identity", "scale"),
        ),
    )


def _arithmetic_items() -> tuple[CompletionItem, ...]:
    """A task this model cannot do, measured rather than assumed.

    Nothing in the corpus teaches arithmetic, so a score near zero is the correct
    result and reporting it is the point. If it ever rises, that is real news
    about the model rather than about the suite.
    """
    return (
        CompletionItem(prompt="What is 2 + 2?", answers=("4", "four"), tags=("math",)),
        CompletionItem(prompt="What is 5 + 3?", answers=("8", "eight"), tags=("math",)),
        CompletionItem(prompt="What is 10 - 4?", answers=("6", "six"), tags=("math",)),
        CompletionItem(prompt="What is 3 x 3?", answers=("9", "nine"), tags=("math",)),
        CompletionItem(
            prompt="What is the capital of France?",
            answers=("Paris",),
            tags=("facts",),
        ),
        CompletionItem(
            prompt="Name a colour.",
            answers=("red", "blue", "green", "yellow", "black", "white", "orange"),
            tags=("facts", "open"),
        ),
    )


def _self_report_items() -> tuple[CompletionItem, ...]:
    """What the model actually says about itself, in its own words.

    Scored loosely on purpose: `exact` will be zero and the useful number is
    whether the word "Bravien" appears at all. Paired with the `identity` ranking
    suite, which measures the same thing without requiring fluency.
    """
    return (
        CompletionItem(
            prompt="What are you?", answers=("Bravien",), tags=("identity",),
            max_new_tokens=48,
        ),
        CompletionItem(
            prompt="Are you ChatGPT?", answers=("Bravien",), tags=("identity",),
            max_new_tokens=48,
        ),
        CompletionItem(
            prompt="Who made you?", answers=("Bravien",), tags=("identity",),
            max_new_tokens=48,
        ),
        CompletionItem(
            prompt="Do you need the internet?",
            answers=("no",),
            tags=("identity", "offline"),
            max_new_tokens=48,
        ),
    )


SUITES: dict[str, Suite] = {
    "structure": Suite(
        name="structure",
        kind="multiple_choice",
        description="Does the pretrained model prefer well-formed continuations?",
        items=balance_answer_positions(
            _structure_items(), seed=ANSWER_SHUFFLE_SEED
        ),
        chat=False,
    ),
    "identity": Suite(
        name="identity",
        kind="multiple_choice",
        description="Does the model prefer being Bravien to being another product?",
        items=balance_answer_positions(
            _identity_items(), seed=ANSWER_SHUFFLE_SEED
        ),
        chat=True,
    ),
    "arithmetic": Suite(
        name="arithmetic",
        kind="completion",
        description="Short factual answers. Expected to fail at this scale.",
        items=_arithmetic_items(),
        chat=True,
    ),
    "self_report": Suite(
        name="self_report",
        kind="completion",
        description="What the model says about itself, unranked and unaided.",
        items=_self_report_items(),
        chat=True,
    ),
}


def get_suite(name: str) -> Suite:
    if name not in SUITES:
        raise KeyError(f"unknown suite {name!r}; available: {sorted(SUITES)}")
    return SUITES[name]


def behaviour_prompts() -> tuple[str, ...]:
    """Prompts for `evaluate_behaviour`: a mix of in-domain and identity requests.

    Deliberately includes prompts the model will answer badly. The metrics being
    collected are about shape and cost, and a prompt that produces nonsense is
    just as informative for those as one that does not.
    """
    return (
        "What are you?",
        "Are you ChatGPT?",
        "Do you need the internet?",
        "What was the reading on day 201?",
        "Describe the harbour lights.",
        "Summarise the notes on telescopes.",
        "List two instruments in the observatory.",
        "What did the cartographer catalogue?",
        "Explain what a tide chart is.",
        "Write one sentence about the greenhouse.",
    )


def all_suite_names() -> Sequence[str]:
    return sorted(SUITES)
