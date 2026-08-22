"""The evaluation run (§27, §28, §71).

One entry point, `evaluate_checkpoint`, which loads nothing and decides nothing
about the model: it takes an engine that already holds a checkpoint, runs the
measurements, and writes a report that says where every number came from.

The report format is the deliverable. It carries:

* the **identity of what was measured** — checkpoint path, step, stage, parameter
  count, tokenizer checksum — so a number can never be attributed to the wrong
  weights;
* the **provenance of the evaluation data**, including whether it was synthetic,
  because a perplexity over generated text is a much weaker claim than one over
  real text and the difference must not be silent (§21, §71);
* the **chance rate and significance** of every benchmark, so a suite result at
  chance reads as a suite result at chance;
* what was **not measured**, listed explicitly. A report that omits its gaps is a
  report that implies it has none (§72).
"""

from __future__ import annotations

import json
import platform
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bravien.evaluation.benchmarks.completion import CompletionReport, run_completion
from bravien.evaluation.benchmarks.multiple_choice import (
    MultipleChoiceReport,
    run_multiple_choice,
)
from bravien.evaluation.benchmarks.suites import (
    EVAL_SYSTEM_PROMPT,
    SUITES,
    Suite,
    behaviour_prompts,
    get_suite,
)
from bravien.evaluation.generation import BehaviourReport, evaluate_behaviour
from bravien.evaluation.perplexity import (
    PerplexityResult,
    evaluate_perplexity,
    random_baseline_perplexity,
)
from bravien.evaluation.safety import (
    SafetyReport,
    evaluate_memorisation,
    evaluate_safety,
    evaluate_template_integrity,
)
from bravien.inference.engine import InferenceEngine
from bravien.utils.logging import get_logger

logger = get_logger("evaluation.evaluator")

#: Seed for the default held-out corpus. Different from any seed used to build
#: training data, which is what makes it held out rather than a training slice.
HELDOUT_SEED = 90_210

#: Everything this repository does not measure yet. Listed in the report rather
#: than left to inference (§72).
NOT_MEASURED: tuple[str, ...] = (
    "instruction following on real (non-synthetic) instruction data",
    "factual accuracy against any external knowledge source",
    "reasoning, coding, or long-context ability",
    "comparison against any hosted model",
    "human preference or helpfulness judgements",
    "adversarial jailbreak resistance",
)


@dataclass
class EvaluationConfig:
    """What to run. Defaults are the full suite at a size that finishes quickly."""

    suites: tuple[str, ...] = ()
    perplexity_documents: int = 64
    behaviour_max_new_tokens: int = 64
    include_behaviour: bool = True
    include_safety: bool = True
    include_memorisation: bool = True
    system_prompt: str | None = EVAL_SYSTEM_PROMPT

    def resolved_suites(self) -> tuple[str, ...]:
        if not self.suites:
            return tuple(sorted(SUITES))
        for name in self.suites:
            get_suite(name)  # raises on an unknown name before anything runs
        return self.suites


@dataclass
class DataProvenance:
    """Where the evaluation text came from (§21)."""

    source: str
    documents: int
    synthetic: bool
    seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "documents": self.documents,
            "synthetic": self.synthetic,
            "seed": self.seed,
            "held_out": True,
        }


@dataclass
class EvaluationReport:
    """Everything one run measured, and everything it did not."""

    model: dict[str, Any] = field(default_factory=dict)
    data: DataProvenance | None = None
    perplexity: PerplexityResult | None = None
    benchmarks: list[MultipleChoiceReport | CompletionReport] = field(
        default_factory=list
    )
    behaviour: BehaviourReport | None = None
    safety: SafetyReport | None = None
    template_integrity: dict[str, Any] = field(default_factory=dict)
    memorisation: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "environment": self.environment,
            "not_measured": list(NOT_MEASURED),
        }
        if self.data is not None:
            payload["evaluation_data"] = self.data.to_dict()
        if self.perplexity is not None:
            payload["perplexity"] = {
                **self.perplexity.to_dict(),
                "uniform_baseline": random_baseline_perplexity(
                    self.perplexity.vocab_size
                ),
            }
        if self.benchmarks:
            payload["benchmarks"] = [b.to_dict() for b in self.benchmarks]
        if self.behaviour is not None:
            payload["behaviour"] = self.behaviour.to_dict()
        if self.safety is not None:
            payload["safety"] = self.safety.to_dict()
        if self.template_integrity:
            payload["template_integrity"] = self.template_integrity
        if self.memorisation:
            payload["memorisation"] = self.memorisation
        return payload

    def save(self, path: str | Path) -> Path:
        """Write the report as JSON."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        logger.info("wrote %s", target)
        return target

    def to_markdown(self) -> str:
        """A human-readable summary, suitable as the basis of a model card (§28).

        Written in the same tone the JSON is: measurements, baselines, and an
        explicit list of what is missing. No adjectives about quality.
        """
        model = self.model
        training = model.get("training", {})
        lines = [
            f"# Evaluation — {model.get('name', 'Bravien')}",
            "",
            "## What was measured",
            "",
            f"- Checkpoint: `{model.get('checkpoint')}`",
            f"- Stage: {training.get('stage')} at step {training.get('step')}, "
            f"{training.get('tokens_seen', 0):,} tokens seen",
            f"- Parameters: {model.get('parameters', 0):,}",
            f"- Context: {model.get('context_length')} tokens, "
            f"vocab {model.get('vocab_size')}",
            f"- Precision: {model.get('precision')} on {model.get('device')}",
            f"- Tokenizer checksum: `{model.get('tokenizer', {}).get('checksum')}`",
            "",
        ]

        if self.data is not None:
            kind = "synthetic" if self.data.synthetic else "real"
            lines += [
                "## Evaluation data",
                "",
                f"- {self.data.documents} held-out documents from "
                f"`{self.data.source}` ({kind})",
                "",
            ]

        if self.perplexity is not None:
            baseline = random_baseline_perplexity(self.perplexity.vocab_size)
            lines += [
                "## Perplexity",
                "",
                f"- {self.perplexity.perplexity:.2f} over "
                f"{self.perplexity.tokens:,} tokens "
                f"({self.perplexity.bits_per_token:.3f} bits/token)",
                f"- Uniform-guessing baseline: {baseline:.0f}",
                "- Comparable only against the same tokenizer "
                f"(`{self.perplexity.tokenizer_checksum[:23]}…`)",
                "",
            ]

        if self.benchmarks:
            lines += ["## Benchmarks", "", "These are repository-specific probe sets, not standard benchmarks.", ""]
            for report in self.benchmarks:
                lines.append(f"- {report.format()}")
            lines.append("")

        if self.behaviour is not None:
            lines += [
                "## Response behaviour",
                "",
                f"- {self.behaviour.format()}",
                "- Measures response shape and cost, not correctness.",
                "",
            ]

        if self.template_integrity:
            held = self.template_integrity.get("markers_cannot_be_forged")
            lines += [
                "## System integrity",
                "",
                f"- Role markers in user content cannot forge a turn: "
                f"{'holds' if held else 'FAILED'} "
                f"({self.template_integrity.get('cases')} cases)",
                "",
            ]

        if self.safety is not None:
            lines += [
                "## Safety behaviour",
                "",
                f"- {self.safety.format()}",
                "- No preference or refusal training stage exists in this "
                "repository, so this is a measurement of its absence.",
                "",
            ]

        if self.memorisation:
            lines += [
                "## Memorisation",
                "",
                f"- Longest verbatim run: "
                f"{self.memorisation.get('longest_verbatim_run')} tokens "
                f"(mean {self.memorisation.get('mean_verbatim_run')}) over "
                f"{self.memorisation.get('documents')} documents",
                f"- {self.memorisation.get('caveat')}",
                "",
            ]

        lines += ["## Not measured", ""]
        lines += [f"- {item}" for item in NOT_MEASURED]
        lines += [
            "",
            "No claim of equivalence to any other model is made or supported by "
            "these numbers.",
            "",
        ]
        return "\n".join(lines)

    def format(self) -> str:
        parts: list[str] = []
        if self.perplexity is not None:
            parts.append(f"ppl {self.perplexity.perplexity:.2f}")
        for report in self.benchmarks:
            if isinstance(report, MultipleChoiceReport):
                parts.append(f"{report.name} {report.accuracy:.0%}")
            else:
                parts.append(f"{report.name} {report.exact_rate:.0%}")
        if self.behaviour is not None:
            parts.append(f"terminated {self.behaviour.termination_rate:.0%}")
        return " | ".join(parts) if parts else "nothing measured"


def _default_heldout_texts(count: int) -> tuple[list[str], DataProvenance]:
    """A held-out slice of the same synthetic corpus the model was trained on.

    Labelled synthetic in the report. A perplexity measured this way says the
    model learned *this generator*, which is a real and much narrower claim than
    "the model models English" (§71).
    """
    from bravien.data.download import generate_seed_corpus

    texts = [doc.text for doc in generate_seed_corpus(count, seed=HELDOUT_SEED)]
    return texts, DataProvenance(
        source="bravien.data.download.generate_seed_corpus",
        documents=len(texts),
        synthetic=True,
        seed=HELDOUT_SEED,
    )


def _run_suite(
    engine: InferenceEngine, suite: Suite, *, system_prompt: str | None
) -> MultipleChoiceReport | CompletionReport:
    if suite.kind == "multiple_choice":
        return run_multiple_choice(
            engine,
            suite.items,  # type: ignore[arg-type]
            name=suite.name,
            chat=suite.chat,
            system_prompt=system_prompt,
        )
    return run_completion(
        engine,
        suite.items,  # type: ignore[arg-type]
        name=suite.name,
        chat=suite.chat,
        system_prompt=system_prompt,
    )


def evaluate_checkpoint(
    engine: InferenceEngine,
    *,
    texts: Sequence[str] | None = None,
    config: EvaluationConfig | None = None,
) -> EvaluationReport:
    """Run the measurements against a loaded engine.

    Args:
        texts: held-out documents for perplexity. Defaults to a slice of the
            synthetic seed corpus generated with an unused seed, recorded as
            synthetic in the report.
    """
    settings = config or EvaluationConfig()
    report = EvaluationReport(
        model=engine.info(),
        environment={
            "python": platform.python_version(),
            "platform": platform.platform(),
            "device": engine.device_info.name,
            "offline": True,
        },
    )

    if texts is None:
        documents, provenance = _default_heldout_texts(settings.perplexity_documents)
    else:
        documents = list(texts)
        provenance = DataProvenance(
            source="caller-supplied", documents=len(documents), synthetic=False
        )
    report.data = provenance

    report.perplexity = evaluate_perplexity(
        engine, documents, max_documents=settings.perplexity_documents
    )

    for name in settings.resolved_suites():
        report.benchmarks.append(
            _run_suite(
                engine, get_suite(name), system_prompt=settings.system_prompt
            )
        )

    if settings.include_behaviour:
        report.behaviour = evaluate_behaviour(
            engine,
            behaviour_prompts(),
            max_new_tokens=settings.behaviour_max_new_tokens,
            system_prompt=settings.system_prompt,
        )

    report.template_integrity = evaluate_template_integrity(engine)

    if settings.include_safety:
        report.safety = evaluate_safety(engine, system_prompt=settings.system_prompt)

    if settings.include_memorisation:
        report.memorisation = evaluate_memorisation(engine, documents[:16])

    logger.info("evaluation complete: %s", report.format())
    return report
