"""Evaluation-harness tests (§27, §71, §19).

This package had no tests, and two defects lived in it that a report reader could
not have detected:

1. `rank_options` broke a tie by taking the lowest index, and every
   multiple-choice item in the repository kept its correct option at index 0. A
   model with no preference at all — identical scores on every option — therefore
   scored **100%**, reported as significant at p≈1.5e-05.
2. Because the answers all sat at index 0, a model that simply always preferred
   the first option scored 100% as well. Neither number measured the model.

The tests below fix the measurement rather than the wording: a degenerate model
must score zero, a first-position-biased model must score at chance, and a suite
that parks its answers at one index must refuse to build. The two regression
tests are `test_a_model_with_no_preference_scores_nothing` and
`test_shipped_suites_spread_their_answers`.

Most tests replace `score_options` — the model-dependent part — and let the real
ranking and aggregation run, so what is under test is the arithmetic that turns
log-probabilities into a reported accuracy. The ones that need real scores build a
genuine engine from a tiny model and a real tokenizer.
"""

from __future__ import annotations

import pytest

from bravien.evaluation import scoring
from bravien.evaluation.benchmarks import multiple_choice as mc
from bravien.evaluation.benchmarks.multiple_choice import (
    MultipleChoiceItem,
    MultipleChoiceOutcome,
    balance_answer_positions,
    binomial_tail,
    run_multiple_choice,
)
from bravien.evaluation.benchmarks.suites import (
    ANSWER_SHUFFLE_SEED,
    SUITES,
    Suite,
)
from bravien.evaluation.scoring import (
    ContinuationScore,
    OptionRanking,
    ScoringError,
    rank_options,
)

# --------------------------------------------------------------------- helpers


def _score(mean_logprob: float, *, tokens: int = 5) -> ContinuationScore:
    """A score with a chosen per-token mean, as the model would have produced."""
    return ContinuationScore(
        logprob=mean_logprob * tokens,
        mean_logprob=mean_logprob,
        tokens=tokens,
        prefix_tokens=9,
    )


def _patch_scores(monkeypatch, fn) -> None:
    """Swap out model scoring, keeping the real ranking and aggregation.

    `rank_options` looks `score_options` up in its own module at call time, so
    patching it here leaves every line of the tie detection, the correctness rule
    and the report arithmetic running for real.
    """
    monkeypatch.setattr(scoring, "score_options", fn)


def _uniform(value: float = -6.928):
    """A model with no preference: every option scores identically.

    This is not a contrived shape — a model whose output distribution is uniform
    over the vocabulary assigns exactly -log(vocab_size) per token to any
    continuation, whatever it says.
    """

    def score_options(engine, prefix, options):
        return [_score(value) for _ in options]

    return score_options


def _prefers_index(index: int):
    """A model that always favours one option position, regardless of content."""

    def score_options(engine, prefix, options):
        return [
            _score(-1.0 if i == index else -5.0) for i in range(len(options))
        ]

    return score_options


def _prefers_substring(needle: str):
    """A model that favours whichever option contains `needle`."""

    def score_options(engine, prefix, options):
        return [
            _score(-1.0 if needle in option else -5.0) for option in options
        ]

    return score_options


ITEMS = (
    MultipleChoiceItem(
        prefix="What are you?",
        options=("I am Bravien.", "I am ChatGPT.", "I am Claude.", "I am Gemini."),
        answer=0,
        tags=("identity",),
    ),
    MultipleChoiceItem(
        prefix="Who made you?",
        options=("Trained here.", "Made by OpenAI.", "Made by Google."),
        answer=0,
        tags=("identity",),
    ),
    MultipleChoiceItem(
        prefix="Do you need the internet?",
        options=("No, I run locally.", "Yes, via an API.", "Yes, I need a key."),
        answer=0,
        tags=("identity", "offline"),
    ),
    MultipleChoiceItem(
        prefix="How big are you?",
        options=("A few million parameters.", "A trillion parameters."),
        answer=0,
        tags=("identity",),
    ),
)


# ------------------------------------------------------ ties are not answers


def test_a_tie_is_reported_rather_than_broken(monkeypatch):
    """The defect at the root of both criticals: `max()` hid an absent preference.

    Every index sharing the top score has to come back, or the caller cannot tell
    "the model chose this" from "the model chose nothing and index 0 won by
    being first".
    """
    _patch_scores(monkeypatch, _uniform())

    ranking = rank_options(None, "prefix", ("a", "b", "c", "d"))

    assert ranking.tied == (0, 1, 2, 3)
    assert ranking.decided is False
    assert ranking.margin == 0.0


def test_a_decided_ranking_names_one_winner(monkeypatch):
    _patch_scores(monkeypatch, _prefers_index(2))

    ranking = rank_options(None, "prefix", ("a", "b", "c", "d"))

    assert ranking.best == 2
    assert ranking.tied == (2,)
    assert ranking.decided is True
    assert ranking.margin == pytest.approx(4.0)


def test_a_tied_item_is_never_correct():
    """Even when the tie includes the right answer, nothing was answered."""
    item = ITEMS[0]
    tied = OptionRanking(
        best=item.answer,
        scores=[_score(-6.9) for _ in item.options],
        tied=tuple(range(len(item.options))),
    )

    outcome = MultipleChoiceOutcome(item=item, ranking=tied)

    assert outcome.predicted == item.answer  # the tie-break landed on it...
    assert outcome.undecided is True
    assert outcome.correct is False  # ...and it still does not count


def test_ties_are_exact_not_approximate(monkeypatch):
    """A close call is a real preference and must not be written off as a tie.

    The tolerance exists for float32 accumulation noise, not for "nearly equal".
    Widening it would start discarding genuine, weakly-held preferences.
    """

    def score_options(engine, prefix, options):
        return [_score(-2.0), _score(-2.0 - 1e-4), _score(-9.0)]

    _patch_scores(monkeypatch, score_options)
    ranking = rank_options(None, "prefix", ("a", "b", "c"))

    assert ranking.decided is True
    assert ranking.tied == (0,)
    assert ranking.margin == pytest.approx(1e-4)


def test_float_noise_still_counts_as_a_tie(monkeypatch):
    def score_options(engine, prefix, options):
        return [_score(-2.0), _score(-2.0 - 1e-12)]

    _patch_scores(monkeypatch, score_options)

    assert rank_options(None, "prefix", ("a", "b")).decided is False


def test_normalisation_changes_the_ranking(monkeypatch):
    """Sum and mean are different measurements; the report names which was used.

    Option 'a' is ten tokens with the better per-token mean; 'b' is two tokens
    with a worse one. The mean prefers 'a', the sum prefers 'b' — because a summed
    log-probability is a sum of negatives and so favours the shorter option
    whatever it says. That is why the mean is the default.
    """

    def score_options(engine, prefix, options):
        return [_score(-1.0, tokens=10), _score(-1.5, tokens=2)]

    _patch_scores(monkeypatch, score_options)

    assert rank_options(None, "p", ("a", "b"), normalise=True).best == 0
    assert rank_options(None, "p", ("a", "b"), normalise=False).best == 1


def test_margin_follows_the_ranking_used(monkeypatch):
    def score_options(engine, prefix, options):
        return [_score(-1.0, tokens=10), _score(-1.5, tokens=2)]

    _patch_scores(monkeypatch, score_options)

    by_mean = rank_options(None, "p", ("a", "b"), normalise=True)
    by_sum = rank_options(None, "p", ("a", "b"), normalise=False)

    assert by_mean.margin == pytest.approx(0.5)  # -1.0 vs -1.5
    assert by_sum.margin == pytest.approx(7.0)  # -3.0 vs -10.0


# ------------------------------------------- what a degenerate model must score


def test_a_model_with_no_preference_scores_nothing(monkeypatch):
    """The regression test for the first critical.

    A model that scores every option identically previously posted 100% on these
    items, because the tie-break took index 0 and index 0 was always right. The
    honest result is zero correct and every item flagged undecided.
    """
    _patch_scores(monkeypatch, _uniform())

    report = run_multiple_choice(None, ITEMS, name="identity")

    assert report.correct == 0
    assert report.accuracy == 0.0
    assert report.undecided == len(ITEMS)
    assert report.above_chance is False
    assert report.mean_margin == 0.0


def test_a_first_position_model_scores_at_chance(monkeypatch):
    """The regression test for the second critical.

    Answers now sit at different indices, so always taking the first option
    scores roughly chance instead of everything. It must land nowhere near 100%,
    and it must not be called significant.
    """
    items = balance_answer_positions(ITEMS, seed=ANSWER_SHUFFLE_SEED)
    _patch_scores(monkeypatch, _prefers_index(0))

    report = run_multiple_choice(None, items, name="identity")

    assert report.undecided == 0  # it did express a preference — just a useless one
    assert report.accuracy < 0.5
    assert report.above_chance is False
    assert set(report.predicted_positions()) == {0}


def test_a_model_that_reads_the_options_scores_well(monkeypatch):
    """The control: the harness must still credit a model that actually answers.

    Without this, "everything scores zero" would look like a fix.
    """
    items = balance_answer_positions(ITEMS, seed=ANSWER_SHUFFLE_SEED)
    # Every correct option in ITEMS is the only one that is not a rival product.
    _patch_scores(monkeypatch, _prefers_substring("Bravien"))
    bravien_items = tuple(i for i in items if any("Bravien" in o for o in i.options))

    report = run_multiple_choice(None, bravien_items, name="identity")

    assert report.correct == len(bravien_items)
    assert report.accuracy == 1.0
    assert report.undecided == 0


def test_the_report_shows_where_predictions_landed(monkeypatch):
    """Position histograms are what let a reader catch this class of artifact."""
    items = balance_answer_positions(ITEMS, seed=ANSWER_SHUFFLE_SEED)
    _patch_scores(monkeypatch, _prefers_index(1))

    payload = run_multiple_choice(None, items, name="identity").to_dict()

    assert payload["predicted_positions"] == {1: len(items)}
    assert len(payload["answer_positions"]) > 1
    assert payload["answer_positions_balanced"] is True
    assert payload["undecided"] == 0


def test_the_summary_line_admits_undecided_items(monkeypatch):
    """A caveat that changes how a number reads has to travel with the number."""
    _patch_scores(monkeypatch, _uniform())

    line = run_multiple_choice(None, ITEMS, name="identity").format()

    assert "0/4" in line
    assert "undecided" in line
    assert "not distinguishable from chance" in line


def test_the_summary_line_flags_an_unbalanced_suite(monkeypatch):
    """Raw items still run — for callers with their own data — but not silently."""
    _patch_scores(monkeypatch, _prefers_index(0))

    line = run_multiple_choice(None, ITEMS, name="raw").format()

    assert "position bias not measurable" in line


def test_per_item_output_records_both_indices(monkeypatch):
    """A reader auditing an item needs the positions, not just the strings."""
    _patch_scores(monkeypatch, _prefers_index(0))
    items = balance_answer_positions(ITEMS, seed=ANSWER_SHUFFLE_SEED)

    report = run_multiple_choice(None, items, name="identity")
    row = report.outcomes[0].to_dict()

    assert row["predicted_index"] == 0
    assert row["answer_index"] == items[0].answer
    assert row["correct"] is (items[0].answer == 0)
    assert row["undecided"] is False
    assert row["tied_options"] == 1


# ------------------------------------------------------- answer position layout


def test_shipped_suites_spread_their_answers():
    """No shipped multiple-choice suite may park its answers at one index.

    Every item in this repository was written answer-first, so before balancing
    all 14 sat at index 0 and no accuracy from them meant anything.
    """
    choice_suites = [s for s in SUITES.values() if s.kind == "multiple_choice"]
    assert choice_suites, "expected at least one multiple-choice suite"

    for suite in choice_suites:
        positions = {item.answer for item in suite.items}
        assert len(positions) > 1, (
            f"suite {suite.name!r} keeps every answer at index {positions}"
        )


def test_balancing_moves_the_index_but_not_the_answer():
    """The permutation must relabel positions without changing the question."""
    balanced = balance_answer_positions(ITEMS, seed=ANSWER_SHUFFLE_SEED)

    assert len(balanced) == len(ITEMS)
    for original, item in zip(ITEMS, balanced):
        assert item.prefix == original.prefix
        assert item.tags == original.tags
        assert sorted(item.options) == sorted(original.options)
        # The correct *text* is preserved; only its index moved.
        assert item.options[item.answer] == original.options[original.answer]


def test_balancing_is_reproducible():
    """Two runs of the same suite must be comparable (§27)."""
    first = balance_answer_positions(ITEMS, seed=ANSWER_SHUFFLE_SEED)
    second = balance_answer_positions(ITEMS, seed=ANSWER_SHUFFLE_SEED)

    assert [i.options for i in first] == [i.options for i in second]
    assert [i.answer for i in first] == [i.answer for i in second]


def test_a_different_seed_gives_a_different_layout():
    same = balance_answer_positions(ITEMS, seed=ANSWER_SHUFFLE_SEED)
    other = balance_answer_positions(ITEMS, seed=ANSWER_SHUFFLE_SEED + 1)

    assert [i.options for i in same] != [i.options for i in other]


def test_editing_one_item_does_not_relayout_the_others():
    """Layout is derived from each item's own prefix, not its list position.

    So adding or removing a probe leaves the rest of the suite exactly as it was,
    and a new report stays comparable to an older one item by item.
    """
    full = balance_answer_positions(ITEMS, seed=ANSWER_SHUFFLE_SEED)
    without_first = balance_answer_positions(ITEMS[1:], seed=ANSWER_SHUFFLE_SEED)

    by_prefix = {item.prefix: item for item in full}
    for item in without_first:
        assert item.options == by_prefix[item.prefix].options
        assert item.answer == by_prefix[item.prefix].answer


def test_a_suite_refuses_to_ship_unbalanced_answers():
    """The guard that stops this critical coming back.

    `ITEMS` is written answer-first, which is how the defect arose in the first
    place, so constructing a suite from it directly must fail.
    """
    with pytest.raises(ValueError, match="every correct answer at index 0"):
        Suite(
            name="unbalanced",
            kind="multiple_choice",
            description="answers all at index 0",
            items=ITEMS,
        )


def test_a_balanced_suite_builds():
    suite = Suite(
        name="balanced",
        kind="multiple_choice",
        description="answers spread across indices",
        items=balance_answer_positions(ITEMS, seed=ANSWER_SHUFFLE_SEED),
    )

    assert len({item.answer for item in suite.items}) > 1


def test_a_single_item_suite_is_exempt():
    """One item cannot be balanced, and rejecting it would be a false positive."""
    suite = Suite(
        name="one",
        kind="multiple_choice",
        description="single item",
        items=(ITEMS[0],),
    )

    assert len(suite.items) == 1


def test_completion_suites_are_not_position_checked():
    """The guard is about ranked options; completion items have no indices."""
    for suite in SUITES.values():
        if suite.kind == "completion":
            assert not hasattr(suite.items[0], "answer")


# ----------------------------------------------------------- item validation


def test_duplicate_options_are_rejected():
    """Identical options are unscoreable: they always tie, by construction."""
    with pytest.raises(ValueError, match="distinct"):
        MultipleChoiceItem(
            prefix="q", options=("same", "same"), answer=0
        )


def test_an_answer_outside_the_options_is_rejected():
    with pytest.raises(ValueError, match="outside"):
        MultipleChoiceItem(prefix="q", options=("a", "b"), answer=5)


def test_one_option_is_not_a_question():
    with pytest.raises(ValueError, match="at least two"):
        MultipleChoiceItem(prefix="q", options=("a",), answer=0)


def test_no_items_is_an_error():
    with pytest.raises(ValueError, match="no items"):
        run_multiple_choice(None, (), name="empty")


# ------------------------------------------------------------- significance


def test_chance_accounts_for_differing_option_counts(monkeypatch):
    """`ITEMS` has 4-, 3-, 3- and 2-option questions, so chance is not 1/4."""
    _patch_scores(monkeypatch, _uniform())
    report = run_multiple_choice(None, ITEMS, name="identity")

    expected = (1 / 4 + 1 / 3 + 1 / 3 + 1 / 2) / 4
    assert report.chance == pytest.approx(expected)


@pytest.mark.parametrize(
    ("successes", "trials", "rate", "expected"),
    [
        (0, 10, 0.25, 1.0),  # "at least none" is certain
        (10, 10, 0.5, 0.5**10),
        (5, 5, 0.25, 0.25**5),
    ],
)
def test_binomial_tail_is_exact(successes, trials, rate, expected):
    assert binomial_tail(successes, trials, rate) == pytest.approx(expected)


def test_significance_depends_on_suite_size_and_chance_rate():
    """A perfect score is not automatically a result.

    Four two-option items scored perfectly give p=0.0625 — above the 0.05 the
    report uses — so `above_chance` stays False no matter how well the model does.
    Four three-option items reach 0.0123 and do clear the bar. Neither number is
    about the model's quality; both are about whether the suite is big enough for
    the accuracy to mean anything (§71).
    """
    assert binomial_tail(4, 4, 1 / 2) == pytest.approx(0.0625)
    assert binomial_tail(4, 4, 1 / 2) > 0.05

    assert binomial_tail(4, 4, 1 / 3) == pytest.approx(1 / 81)
    assert binomial_tail(4, 4, 1 / 3) < 0.05


# --------------------------------------------------- real scores, real engine


@pytest.fixture(scope="session")
def eval_engine(trained_tokenizer):
    """A real engine: untrained weights, but a real tokenizer and real forward pass.

    Untrained is the point for these tests — they check the scoring arithmetic,
    which must be right regardless of what the model knows.
    """
    import torch

    from bravien.inference.engine import EngineConfig, InferenceEngine
    from bravien.model.config import BravienConfig
    from bravien.model.model import BravienForCausalLM

    config = BravienConfig(
        vocab_size=trained_tokenizer.vocab_size,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
        dropout=0.0,
        attention_dropout=0.0,
    )
    torch.manual_seed(0)
    model = BravienForCausalLM(config)
    return InferenceEngine(
        model,
        trained_tokenizer,
        config=EngineConfig(device="cpu", dtype="fp32"),
    )


def test_scoring_matches_the_models_own_logits(eval_engine):
    """The documented off-by-one, verified against a hand-computed value.

    `logits[t]` predicts `ids[t+1]`, so a continuation starting at absolute index
    `p` is scored from `logits[p-1:-1]`. Getting this wrong yields a plausible
    number that ranks options by the wrong thing, which no downstream assertion
    would catch.
    """
    import torch

    prefix, continuation = "The reading was", " 30.25 grams"
    result = scoring.score_continuation(eval_engine, prefix, continuation)

    tokenizer = eval_engine.tokenizer
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    continuation_ids = tokenizer.encode(continuation, add_special_tokens=False)
    ids = torch.tensor([prefix_ids + continuation_ids], dtype=torch.long)

    with torch.inference_mode():
        logits = eval_engine.model(input_ids=ids).logits.to(torch.float32)

    expected = 0.0
    for offset, target in enumerate(continuation_ids):
        step = len(prefix_ids) - 1 + offset
        expected += float(
            torch.log_softmax(logits[0, step], dim=-1)[target].item()
        )

    assert result.tokens == len(continuation_ids)
    assert result.prefix_tokens == len(prefix_ids)
    assert result.logprob == pytest.approx(expected, abs=1e-4)
    assert result.mean_logprob == pytest.approx(
        expected / len(continuation_ids), abs=1e-4
    )


def test_scoring_is_deterministic(eval_engine):
    """Nothing here samples, so repeated scoring must be bit-for-bit stable (§27)."""
    first = scoring.score_continuation(eval_engine, "Notes on", " telescopes.")
    second = scoring.score_continuation(eval_engine, "Notes on", " telescopes.")

    assert first == second


def test_an_untrained_model_is_near_the_uniform_baseline(eval_engine):
    """A sanity floor on the scores themselves.

    Untrained weights should sit near -log(vocab_size) per token. A score far
    from that would mean the scoring path, not the model, is wrong.
    """
    import math

    result = scoring.score_continuation(eval_engine, "Notes on", " telescopes.")
    uniform = -math.log(eval_engine.tokenizer.vocab_size)

    assert result.mean_logprob == pytest.approx(uniform, abs=1.5)


def test_a_bare_continuation_is_scored_against_bos(eval_engine):
    """Position 0 has no predictor, so an empty prefix must not lose a token."""
    result = scoring.score_continuation(eval_engine, "", " telescopes.")

    assert result.prefix_tokens == 1
    assert result.tokens == len(
        eval_engine.tokenizer.encode(" telescopes.", add_special_tokens=False)
    )


def test_an_empty_continuation_is_an_error(eval_engine):
    with pytest.raises(ScoringError, match="zero tokens"):
        scoring.score_continuation(eval_engine, "prefix", "")


def test_a_continuation_too_long_for_the_context_is_an_error(eval_engine):
    """Truncating the continuation would change the question, so it must raise."""
    huge = "telescopes and harbour lights " * 200

    with pytest.raises(ScoringError, match="does not fit"):
        scoring.score_continuation(eval_engine, "prefix", huge)


def test_an_overlong_prefix_is_cut_and_the_cut_is_reported(eval_engine):
    """Dropping old context changes the conditioning, so it is flagged, not hidden."""
    long_prefix = "The reading was 30.25 grams on day 201. " * 40

    result = scoring.score_continuation(eval_engine, long_prefix, " Notes.")

    assert result.truncated_prefix is True
    assert result.prefix_tokens + result.tokens <= eval_engine.max_context


def test_real_ranking_reaches_a_verdict(eval_engine):
    """End to end with real scores: a ranking is produced and it is internally consistent."""
    ranking = rank_options(
        eval_engine,
        "The reading was",
        (" 30.25 grams on day 201.", " the the the.", " canal described beside."),
    )

    assert 0 <= ranking.best < 3
    assert len(ranking.scores) == 3
    assert ranking.best in ranking.tied
    assert ranking.margin >= 0.0
    if ranking.decided:
        key = ranking.scores[ranking.best].mean_logprob
        assert all(
            key >= s.mean_logprob for s in ranking.scores
        ), "the reported winner must hold the top score"
