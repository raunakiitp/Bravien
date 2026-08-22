"""Benchmark suites (§27).

Two scoring methods, deliberately kept apart:

* `multiple_choice` ranks fixed options by log-probability. It needs no decoding,
  so it is fast, deterministic, and measurable on a model far too weak to
  generate a coherent answer — which is exactly the situation here.
* `completion` decodes greedily and compares strings. It is the harder test and
  the one whose failure is most informative.

`suites` holds the built-in probe sets. They are small and specific to this
repository's corpus; they are not standard benchmarks and the reports say so
(§71).
"""

from __future__ import annotations

from bravien.evaluation.benchmarks.completion import (
    CompletionItem,
    CompletionOutcome,
    CompletionReport,
    normalise_answer,
    run_completion,
)
from bravien.evaluation.benchmarks.multiple_choice import (
    MultipleChoiceItem,
    MultipleChoiceOutcome,
    MultipleChoiceReport,
    binomial_tail,
    run_multiple_choice,
)
from bravien.evaluation.benchmarks.suites import (
    EVAL_SYSTEM_PROMPT,
    SUITES,
    Suite,
    all_suite_names,
    behaviour_prompts,
    get_suite,
)

__all__ = [
    "EVAL_SYSTEM_PROMPT",
    "SUITES",
    "CompletionItem",
    "CompletionOutcome",
    "CompletionReport",
    "MultipleChoiceItem",
    "MultipleChoiceOutcome",
    "MultipleChoiceReport",
    "Suite",
    "all_suite_names",
    "behaviour_prompts",
    "binomial_tail",
    "get_suite",
    "normalise_answer",
    "run_completion",
    "run_multiple_choice",
]
