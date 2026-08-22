"""Safety measurements (§27, §53, §72).

The distinction this package is built around: some safety properties belong to the
**system** and some to the **model**, and they are not measured the same way.

*System* properties are deterministic and this repository does hold them today —
content cannot forge a turn boundary, a request cannot widen its own token cap,
the serving path makes no network call. Those are checked, and they pass.

*Model* behaviour — declining a harmful request — has had no training stage in
this repository at all. There is no preference or refusal tuning, so the honest
measurement is a number near zero, reported as a number near zero. Presenting a
3M-parameter checkpoint as aligned because a suite was run would be exactly the
pretence §72 forbids.
"""

from __future__ import annotations

from bravien.evaluation.safety.probes import (
    DEFAULT_PROBES,
    REFUSAL_MARKERS,
    SafetyOutcome,
    SafetyProbe,
    SafetyReport,
    evaluate_memorisation,
    evaluate_safety,
    evaluate_template_integrity,
    looks_like_refusal,
)

__all__ = [
    "DEFAULT_PROBES",
    "REFUSAL_MARKERS",
    "SafetyOutcome",
    "SafetyProbe",
    "SafetyReport",
    "evaluate_memorisation",
    "evaluate_safety",
    "evaluate_template_integrity",
    "looks_like_refusal",
]
