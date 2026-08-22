"""Bravien's special tokens.

The ids are fixed by the order of `SPECIAL_TOKENS` and are reserved before any
learned merge, so token 0 is always PAD in every Bravien tokenizer ever trained.
Changing this order breaks every existing checkpoint.
"""

from __future__ import annotations

PAD = "<PAD>"
UNK = "<UNK>"
BOS = "<BOS>"
EOS = "<EOS>"

SYSTEM_OPEN = "<SYSTEM>"
SYSTEM_CLOSE = "</SYSTEM>"
USER_OPEN = "<USER>"
USER_CLOSE = "</USER>"
ASSISTANT_OPEN = "<ASSISTANT>"
ASSISTANT_CLOSE = "</ASSISTANT>"
TOOL_OPEN = "<TOOL>"
TOOL_CLOSE = "</TOOL>"

#: Order defines token ids 0..n-1. Append only; never reorder or remove.
SPECIAL_TOKENS: list[str] = [
    PAD,
    UNK,
    BOS,
    EOS,
    SYSTEM_OPEN,
    SYSTEM_CLOSE,
    USER_OPEN,
    USER_CLOSE,
    ASSISTANT_OPEN,
    ASSISTANT_CLOSE,
    TOOL_OPEN,
    TOOL_CLOSE,
]

SPECIAL_TOKEN_IDS: dict[str, int] = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}

PAD_ID = SPECIAL_TOKEN_IDS[PAD]
UNK_ID = SPECIAL_TOKEN_IDS[UNK]
BOS_ID = SPECIAL_TOKEN_IDS[BOS]
EOS_ID = SPECIAL_TOKEN_IDS[EOS]

#: Markers that must never survive inside user-supplied content. See
#: `bravien.tokenizer.templates.sanitize_content`.
ROLE_MARKERS: tuple[str, ...] = (
    SYSTEM_OPEN,
    SYSTEM_CLOSE,
    USER_OPEN,
    USER_CLOSE,
    ASSISTANT_OPEN,
    ASSISTANT_CLOSE,
    TOOL_OPEN,
    TOOL_CLOSE,
    BOS,
    EOS,
    PAD,
)
