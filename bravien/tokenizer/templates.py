"""The Bravien chat template (§25).

This module is the *only* place the conversation wire format is defined. The
trainer, the inference engine, and the API all go through `format_conversation`,
so the text the model sees at inference is byte-identical to what it saw during
instruction tuning. Drift between those two is the most common cause of a
fine-tuned model behaving worse than its evaluation suggested.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal

from bravien.tokenizer.special_tokens import (
    ASSISTANT_CLOSE,
    ASSISTANT_OPEN,
    BOS,
    EOS,
    ROLE_MARKERS,
    SYSTEM_CLOSE,
    SYSTEM_OPEN,
    TOOL_CLOSE,
    TOOL_OPEN,
    USER_CLOSE,
    USER_OPEN,
)

Role = Literal["system", "user", "assistant", "tool"]

_OPEN: dict[str, str] = {
    "system": SYSTEM_OPEN,
    "user": USER_OPEN,
    "assistant": ASSISTANT_OPEN,
    "tool": TOOL_OPEN,
}
_CLOSE: dict[str, str] = {
    "system": SYSTEM_CLOSE,
    "user": USER_CLOSE,
    "assistant": ASSISTANT_CLOSE,
    "tool": TOOL_CLOSE,
}


@dataclass(frozen=True)
class Segment:
    """A run of template text plus whether loss should be computed on it.

    `trainable` is what makes supervised fine-tuning correct: the prompt is
    context the model conditions on, and only the assistant's own words are
    targets it should be penalised for getting wrong.
    """

    text: str
    trainable: bool


class ChatTemplateError(ValueError):
    pass


def sanitize_content(text: str) -> str:
    """Neutralise role markers appearing inside message content.

    Without this, a user could type `</USER><ASSISTANT>` and forge a turn
    boundary — the text-level equivalent of SQL injection. Markers are defanged
    by breaking them with a zero-width-free visible substitution so the content
    stays readable but no longer tokenises to a special id.
    """
    for marker in ROLE_MARKERS:
        if marker in text:
            text = text.replace(marker, marker.replace("<", "‹").replace(">", "›"))
    return text


def normalize_messages(messages: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    """Validate roles and coerce content to strings."""
    out: list[dict[str, str]] = []
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise ChatTemplateError(f"message {i} is not an object: {msg!r}")
        role = msg.get("role")
        if role not in _OPEN:
            raise ChatTemplateError(
                f"message {i} has unsupported role {role!r}; "
                f"expected one of {sorted(_OPEN)}"
            )
        content = msg.get("content")
        if content is None:
            raise ChatTemplateError(f"message {i} (role={role}) has no content")
        out.append({"role": role, "content": str(content)})
    return out


def format_conversation(
    messages: Iterable[dict[str, Any]],
    *,
    add_generation_prompt: bool = False,
    add_bos: bool = True,
    add_eos: bool = False,
) -> list[Segment]:
    """Render a conversation into segments.

    Args:
        add_generation_prompt: append a bare `<ASSISTANT>` so the model
            continues as the assistant. Used at inference, not during training.
        add_eos: append `<EOS>` after the final turn. Used during training so
            the model learns where to stop.
    """
    msgs = normalize_messages(messages)
    segments: list[Segment] = []

    if add_bos:
        segments.append(Segment(BOS, trainable=False))

    for msg in msgs:
        role, content = msg["role"], sanitize_content(msg["content"])
        if role == "assistant":
            # Marker and content are separate segments so the opening tag stays
            # untrained (it is the cue we supply) while the reply is a target.
            segments.append(Segment(_OPEN[role], trainable=False))
            segments.append(Segment(content, trainable=True))
            segments.append(Segment(_CLOSE[role], trainable=True))
        else:
            segments.append(
                Segment(f"{_OPEN[role]}{content}{_CLOSE[role]}", trainable=False)
            )

    if add_generation_prompt:
        if msgs and msgs[-1]["role"] == "assistant":
            raise ChatTemplateError(
                "add_generation_prompt=True but the last message is already "
                "from the assistant"
            )
        segments.append(Segment(ASSISTANT_OPEN, trainable=False))

    if add_eos:
        segments.append(Segment(EOS, trainable=True))

    return segments


def render(segments: Iterable[Segment]) -> str:
    """Flatten segments to the exact string the tokenizer will see."""
    return "".join(seg.text for seg in segments)


def format_prompt(messages: Iterable[dict[str, Any]]) -> str:
    """Convenience: the inference-time prompt string for a conversation."""
    return render(format_conversation(messages, add_generation_prompt=True))
