"""Context-window budgeting against the real tokenizer (§Phase 7).

The context window is a hard property of the weights: `max_position_embeddings`
is the largest position the RoPE tables and the KV cache were built for, and
exceeding it raises rather than degrades. So every request has to be made to fit
*before* it reaches the model, and the only honest way to know whether it fits is
to tokenize it with the tokenizer the checkpoint was trained with.

Two rules drive everything here:

* **Count, never estimate.** A byte-level BPE tokenizer on English prose runs
  around 1.5 characters per token, but that ratio is a property of the corpus,
  not a constant — code, CJK, and long words all move it. A character-based
  divisor that happens to be wrong in the optimistic direction produces a prompt
  that is silently cut to a fraction of itself, which is exactly the failure this
  module exists to prevent.
* **Drop in a defined order, and say what was dropped.** Truncation is lossy, so
  the loss is applied where it costs least (the oldest history) and reported to
  the caller instead of being hidden in a log line nobody reads.

The priority when something must go, in order of what is kept:

1. system messages — Bravien's identity and safety rules
2. the most recent user message — the question actually being asked
3. recent conversation turns, newest first
4. older history

`plan_context` is pure: it takes messages and returns the messages that fit plus
the accounting. It never talks to a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from bravien.tokenizer.templates import (
    format_conversation,
    normalize_messages,
    render,
)
from bravien.tokenizer.tokenizer import BravienTokenizer
from bravien.utils.logging import get_logger

logger = get_logger("inference.context")


class ContextOverflowError(ValueError):
    """The reservation leaves no room for a prompt at all.

    Distinct from "the prompt was too long": this means the *configuration* is
    impossible, so no input could ever have fitted, and the caller needs to
    reserve fewer output tokens rather than send less text.
    """


@dataclass(frozen=True)
class ContextPlan:
    """What fits, and what it cost to make it fit.

    Every field is a measured token count from the real tokenizer. `messages` is
    what should actually be sent; the counts describe how it was derived.
    """

    #: The conversation that fits, in original order.
    messages: tuple[dict[str, str], ...]
    #: Exact token length of the rendered prompt for `messages`.
    input_tokens: int
    #: The model's context window.
    max_context_tokens: int
    #: Tokens held back for the answer.
    reserved_output_tokens: int
    #: Whole messages removed to make room.
    truncated_turns: int
    #: A system message had to be shortened. Should be treated as a misconfiguration.
    system_truncated: bool
    #: The newest user message had to be shortened.
    latest_user_truncated: bool
    #: Token cost of the template scaffolding (BOS plus the generation cue).
    overhead_tokens: int

    @property
    def truncated(self) -> bool:
        return bool(
            self.truncated_turns or self.system_truncated or self.latest_user_truncated
        )

    @property
    def available_tokens(self) -> int:
        """Room the prompt was allowed to occupy."""
        return self.max_context_tokens - self.reserved_output_tokens

    def to_dict(self) -> dict[str, Any]:
        """The accounting, for the API and the UI. Excludes message content."""
        return {
            "input_tokens": self.input_tokens,
            "max_context_tokens": self.max_context_tokens,
            "reserved_output_tokens": self.reserved_output_tokens,
            "available_tokens": self.available_tokens,
            "overhead_tokens": self.overhead_tokens,
            "truncated": self.truncated,
            "truncated_turns": self.truncated_turns,
            "system_truncated": self.system_truncated,
            "latest_user_truncated": self.latest_user_truncated,
        }


def _message_tokens(tokenizer: BravienTokenizer, message: dict[str, str]) -> int:
    """Exact token cost of one message including its role markers.

    Routed through `format_conversation` so this cannot drift from the string the
    model is actually given. Per-message counts sum to the whole-prompt count
    because every role marker is a hard split for the pre-tokenizer, so no BPE
    merge spans a message boundary.
    """
    text = render(format_conversation([message], add_bos=False))
    return len(tokenizer.encode(text))


def _overhead_tokens(tokenizer: BravienTokenizer) -> int:
    """Cost of the scaffolding around the messages: BOS and the generation cue."""
    text = render(
        format_conversation([], add_bos=True, add_generation_prompt=True)
    )
    return len(tokenizer.encode(text))


def _fit_content(
    tokenizer: BravienTokenizer,
    message: dict[str, str],
    budget: int,
    *,
    keep: str,
) -> tuple[dict[str, str], bool]:
    """Shorten one message's content until the whole message fits in `budget`.

    `keep` picks which end survives: "head" for a system prompt, whose identity
    and rules lead, and "tail" for a user message, whose actual question
    usually trails. Returns the message unchanged when it already fits.

    Binary search on a character count rather than on tokens: token boundaries
    are not monotonic in a way that makes an incremental walk safe, and the
    content is re-encoded for the true cost on every probe anyway.
    """
    if budget <= 0:
        return {**message, "content": ""}, True
    if _message_tokens(tokenizer, message) <= budget:
        return message, False

    content = message["content"]
    lo, hi = 0, len(content)
    best = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = content[:mid] if keep == "head" else content[len(content) - mid :]
        if _message_tokens(tokenizer, {**message, "content": candidate}) <= budget:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1

    return {**message, "content": best}, True


def plan_context(
    tokenizer: BravienTokenizer,
    messages: Iterable[dict[str, Any]],
    *,
    max_context: int,
    reserved_output: int,
) -> ContextPlan:
    """Select the messages that fit the context window, and report the cost.

    Raises:
        ContextOverflowError: if `reserved_output` leaves no room for any prompt.
        ChatTemplateError: if a message has an unknown role or no content.
    """
    if max_context <= 0:
        raise ContextOverflowError(f"max_context must be positive, got {max_context}")
    if reserved_output < 0:
        raise ContextOverflowError(
            f"reserved_output must not be negative, got {reserved_output}"
        )

    msgs = normalize_messages(messages)
    overhead = _overhead_tokens(tokenizer)
    budget = max_context - reserved_output - overhead

    if budget <= 0:
        raise ContextOverflowError(
            f"reserving {reserved_output} output tokens of a {max_context}-token "
            f"context leaves {budget} tokens for the prompt after {overhead} "
            f"tokens of template overhead; reserve fewer output tokens"
        )

    if not msgs:
        return ContextPlan(
            messages=(),
            input_tokens=overhead,
            max_context_tokens=max_context,
            reserved_output_tokens=reserved_output,
            truncated_turns=0,
            system_truncated=False,
            latest_user_truncated=False,
            overhead_tokens=overhead,
        )

    costs = [_message_tokens(tokenizer, m) for m in msgs]
    system_idx = [i for i, m in enumerate(msgs) if m["role"] == "system"]
    user_idx = [i for i, m in enumerate(msgs) if m["role"] == "user"]
    latest_user = user_idx[-1] if user_idx else None

    kept: dict[int, dict[str, str]] = {}
    system_truncated = False
    latest_user_truncated = False
    spent = 0

    # Divide the budget between the system prompt and the newest user message
    # before spending any of it. Priority is an ordering, not a licence for the
    # first item to consume everything: a system prompt that grew larger than the
    # context would otherwise eat the whole budget and leave the user's actual
    # question truncated to nothing, which preserves the letter of "keep the
    # system message" while destroying the point of the request.
    user_need = costs[latest_user] if latest_user is not None else 0
    system_need = sum(costs[i] for i in system_idx)

    if system_need + user_need <= budget:
        system_allow = system_need
    else:
        # Under contention neither side may starve the other, so the system
        # prompt is capped at half the room; it then gets back any share the
        # user message does not need.
        system_allow = min(system_need, max(1, budget // 2))
        slack = budget - system_allow - user_need
        if slack > 0:
            system_allow = min(system_need, system_allow + slack)

    # 1. System messages. Kept whole if at all possible: dropping Bravien's
    #    identity and safety rules to make room for chat history is never the
    #    right trade, so these are shortened rather than removed, and loudly.
    for i in system_idx:
        remaining = system_allow - spent
        if costs[i] <= remaining:
            kept[i] = msgs[i]
            spent += costs[i]
            continue
        fitted, was_cut = _fit_content(
            tokenizer, msgs[i], remaining, keep="head"
        )
        kept[i] = fitted
        spent += _message_tokens(tokenizer, fitted)
        system_truncated = system_truncated or was_cut
        logger.warning(
            "system message of %d tokens shortened to %d: it does not fit a "
            "%d-token context with %d tokens reserved for output. Use a system "
            "prompt sized for this model.",
            costs[i],
            _message_tokens(tokenizer, fitted),
            max_context,
            reserved_output,
        )

    # 2. The newest user message: the question actually being asked.
    if latest_user is not None and latest_user not in kept:
        remaining = budget - spent
        if costs[latest_user] <= remaining:
            kept[latest_user] = msgs[latest_user]
            spent += costs[latest_user]
        else:
            fitted, was_cut = _fit_content(
                tokenizer, msgs[latest_user], remaining, keep="tail"
            )
            kept[latest_user] = fitted
            spent += _message_tokens(tokenizer, fitted)
            latest_user_truncated = was_cut
            logger.warning(
                "newest user message of %d tokens shortened to fit %d remaining "
                "tokens",
                costs[latest_user],
                remaining,
            )

    # 3 and 4. Remaining history, newest first, whole messages only. A partial
    #    older turn is worth less than the tokens it costs, and a half-sentence
    #    attributed to the user or the assistant is misleading context.
    for i in range(len(msgs) - 1, -1, -1):
        if i in kept:
            continue
        if spent + costs[i] <= budget:
            kept[i] = msgs[i]
            spent += costs[i]

    ordered = tuple(kept[i] for i in sorted(kept))
    dropped = len(msgs) - len(ordered)

    # Measure the assembled prompt rather than trusting the running total.
    input_tokens = overhead + sum(_message_tokens(tokenizer, m) for m in ordered)

    if dropped:
        logger.info(
            "context: dropped %d of %d messages to fit %d tokens (%d reserved "
            "for output)",
            dropped,
            len(msgs),
            budget,
            reserved_output,
        )

    return ContextPlan(
        messages=ordered,
        input_tokens=input_tokens,
        max_context_tokens=max_context,
        reserved_output_tokens=reserved_output,
        truncated_turns=dropped,
        system_truncated=system_truncated,
        latest_user_truncated=latest_user_truncated,
        overhead_tokens=overhead,
    )
