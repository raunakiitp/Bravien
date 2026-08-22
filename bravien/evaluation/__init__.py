"""Measurement, so that nothing has to be claimed (§27, §71, §72).

A number is only allowed to mean something if it comes with what it was measured
against. That principle shapes every module here:

* `scoring` — conditional log-probability, the primitive the rest is built on.
* `perplexity` — held-out loss, stamped with the tokenizer checksum that makes it
  comparable, and never comparable without it.
* `generation` — whether the model stops, repeats, or emits nothing. Shape and
  cost, explicitly not correctness.
* `benchmarks` — probe sets with chance rates and exact binomial significance, so
  a result at chance reads as a result at chance.
* `safety` — deterministic system properties, which hold, kept separate from model
  refusal behaviour, which has had no training stage here.
* `evaluator` — one run over all of it, producing a JSON report and a markdown
  summary stamped with the identity of the checkpoint measured.

The point of the package is §71: no claim about this model — including a
comparison to any other model — is supported by anything except a number produced
here, over stated data, with its baseline next to it.
"""

from __future__ import annotations

from bravien.evaluation.benchmarks import (
    EVAL_SYSTEM_PROMPT,
    SUITES,
    CompletionItem,
    CompletionReport,
    MultipleChoiceItem,
    MultipleChoiceReport,
    Suite,
    all_suite_names,
    behaviour_prompts,
    binomial_tail,
    get_suite,
    normalise_answer,
    run_completion,
    run_multiple_choice,
)
from bravien.evaluation.evaluator import (
    NOT_MEASURED,
    DataProvenance,
    EvaluationConfig,
    EvaluationReport,
    evaluate_checkpoint,
)
from bravien.evaluation.generation import (
    DETERMINISTIC,
    BehaviourReport,
    GenerationSample,
    distinct_ngram_ratio,
    evaluate_behaviour,
    leaks_role_markers,
    longest_repeat_run,
)
from bravien.evaluation.perplexity import (
    PerplexityResult,
    evaluate_perplexity,
    random_baseline_perplexity,
)
from bravien.evaluation.safety import (
    DEFAULT_PROBES,
    SafetyProbe,
    SafetyReport,
    evaluate_memorisation,
    evaluate_safety,
    evaluate_template_integrity,
    looks_like_refusal,
)
from bravien.evaluation.scoring import (
    ContinuationScore,
    ScoringError,
    rank_options,
    score_continuation,
    score_options,
)

__all__ = [
    "DEFAULT_PROBES",
    "DETERMINISTIC",
    "EVAL_SYSTEM_PROMPT",
    "NOT_MEASURED",
    "SUITES",
    "BehaviourReport",
    "CompletionItem",
    "CompletionReport",
    "ContinuationScore",
    "DataProvenance",
    "EvaluationConfig",
    "EvaluationReport",
    "GenerationSample",
    "MultipleChoiceItem",
    "MultipleChoiceReport",
    "PerplexityResult",
    "SafetyProbe",
    "SafetyReport",
    "ScoringError",
    "Suite",
    "all_suite_names",
    "behaviour_prompts",
    "binomial_tail",
    "distinct_ngram_ratio",
    "evaluate_behaviour",
    "evaluate_checkpoint",
    "evaluate_memorisation",
    "evaluate_perplexity",
    "evaluate_safety",
    "evaluate_template_integrity",
    "get_suite",
    "leaks_role_markers",
    "longest_repeat_run",
    "looks_like_refusal",
    "normalise_answer",
    "random_baseline_perplexity",
    "rank_options",
    "run_completion",
    "run_multiple_choice",
    "score_continuation",
    "score_options",
]
