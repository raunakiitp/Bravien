"""Sequence packing (§12, stage 6).

Two objectives, deliberately handled differently:

* **Pretraining** concatenates documents and chops the stream into fixed-length
  blocks. Every position is a training target, so there is no padding waste. A
  block may straddle a document boundary; the EOS token written between
  documents is what teaches the model that a boundary happened.

* **Instruction tuning** keeps each conversation intact, because supervision is
  masked to assistant spans only and a conversation split across two sequences
  would be trained against a prompt it cannot see. Batches are padded to their
  own longest member, not to a global maximum.

Multi-conversation packing with block-diagonal attention is *not* implemented:
it needs per-pair masks throughout the attention stack, and padding is correct
and cheap enough at this scale (§72 — no pretending).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from bravien.tokenizer.special_tokens import SPECIAL_TOKEN_IDS
from bravien.tokenizer.tokenizer import IGNORE_INDEX

#: The reserved ids as a set. `SPECIAL_TOKEN_IDS` is keyed by token *text*, so
#: testing an id against it directly always misses.
_SPECIAL_IDS = frozenset(SPECIAL_TOKEN_IDS.values())

if TYPE_CHECKING:
    from bravien.tokenizer.tokenizer import BravienTokenizer


def pack_sequences(
    tokens: Iterable[int], seq_len: int, *, drop_last: bool = True
) -> Iterator[list[int]]:
    """Chop a token stream into fixed-length blocks.

    Args:
        drop_last: discard a trailing partial block. Default True: a short final
            block would need padding, and one ragged sequence per corpus is not
            worth the special case.
    """
    if seq_len <= 1:
        raise ValueError("seq_len must be greater than 1")

    block: list[int] = []
    for token in tokens:
        block.append(token)
        if len(block) == seq_len:
            yield block
            block = []
    if block and not drop_last:
        yield block


def count_packed_sequences(total_tokens: int, seq_len: int) -> int:
    """How many training sequences a token stream yields.

    One extra token is needed beyond each block so every position has a next-token
    target, which is why this is not simply `total_tokens // seq_len`.
    """
    usable = total_tokens - 1
    return max(usable // seq_len, 0)


# ------------------------------------------------------------ instruction data


@dataclass
class SFTExample:
    """One supervised conversation, already tokenized.

    `labels` is IGNORE_INDEX everywhere the model must not be trained to
    reproduce — system prompts, user turns, and padding (§18).
    """

    input_ids: list[int]
    labels: list[int]
    source: str = ""
    meta: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.input_ids) != len(self.labels):
            raise ValueError(
                f"input_ids ({len(self.input_ids)}) and labels "
                f"({len(self.labels)}) must be the same length"
            )

    def __len__(self) -> int:
        return len(self.input_ids)

    @property
    def num_supervised(self) -> int:
        return sum(1 for label in self.labels if label != IGNORE_INDEX)


class SFTPackingError(ValueError):
    pass


def build_sft_example(
    messages: Sequence[dict[str, str]],
    tokenizer: BravienTokenizer,
    *,
    max_length: int,
    source: str = "",
    add_bos: bool = True,
) -> SFTExample | None:
    """Tokenize one conversation into an input/label pair.

    Returns None when the example would teach nothing useful. Two cases: a
    conversation with no assistant turn, which carries no supervision at all; and
    one truncated so hard that the only supervised tokens left are the closing
    markers. The second is worth dropping rather than keeping — its whole lesson
    is "stop immediately, regardless of what was asked".
    """
    encoding = tokenizer.encode_chat(
        messages, add_generation_prompt=False, add_bos=add_bos, add_eos=True
    )
    input_ids = list(encoding.input_ids)
    labels = list(encoding.labels)

    if len(input_ids) > max_length:
        # Truncate from the *left*: the final assistant turn is the supervision,
        # so dropping early context preserves something trainable, while
        # truncating from the right would remove the answer itself.
        input_ids = input_ids[-max_length:]
        labels = labels[-max_length:]

    # Special ids are excluded deliberately. `<EOS>` is always supervised and
    # always last, so counting it would make every example look supervised and
    # this guard would never fire.
    if not any(
        label != IGNORE_INDEX and label not in _SPECIAL_IDS for label in labels
    ):
        return None

    return SFTExample(input_ids=input_ids, labels=labels, source=source)


@dataclass
class SFTPackStats:
    conversations: int = 0
    kept: int = 0
    dropped_unsupervised: int = 0
    truncated: int = 0
    total_tokens: int = 0
    supervised_tokens: int = 0

    @property
    def supervision_rate(self) -> float:
        return self.supervised_tokens / self.total_tokens if self.total_tokens else 0.0


def build_sft_examples(
    conversations: Iterable[Sequence[dict[str, str]]],
    tokenizer: BravienTokenizer,
    *,
    max_length: int,
    source: str = "",
    stats: SFTPackStats | None = None,
) -> Iterator[SFTExample]:
    """Tokenize many conversations, tracking how much supervision survived."""
    st = stats if stats is not None else SFTPackStats()

    for messages in conversations:
        st.conversations += 1
        example = build_sft_example(
            messages, tokenizer, max_length=max_length, source=source
        )
        if example is None:
            st.dropped_unsupervised += 1
            continue
        if len(example) == max_length:
            st.truncated += 1
        st.kept += 1
        st.total_tokens += len(example)
        st.supervised_tokens += example.num_supervised
        yield example


def collate_sft(
    batch: Sequence[SFTExample], pad_id: int, *, pad_to_multiple_of: int = 8
) -> dict[str, torch.Tensor]:
    """Pad a batch to its own longest member.

    Args:
        pad_to_multiple_of: round the padded length up so matmul shapes stay
            tensor-core friendly. Costs a few padding tokens, which are masked
            out of both attention and loss anyway.

    Returns:
        `input_ids`, `labels`, and a boolean `attention_mask` where True means
        "real token".
    """
    if not batch:
        raise SFTPackingError("cannot collate an empty batch")

    longest = max(len(ex) for ex in batch)
    if pad_to_multiple_of > 1:
        remainder = longest % pad_to_multiple_of
        if remainder:
            longest += pad_to_multiple_of - remainder

    input_ids = torch.full((len(batch), longest), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), longest), IGNORE_INDEX, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), longest), dtype=torch.bool)

    for row, example in enumerate(batch):
        n = len(example)
        input_ids[row, :n] = torch.tensor(example.input_ids, dtype=torch.long)
        labels[row, :n] = torch.tensor(example.labels, dtype=torch.long)
        attention_mask[row, :n] = True

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }
