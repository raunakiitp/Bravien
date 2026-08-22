"""Torch datasets over prepared data (§12, §15).

Reading strategy: the tokenized corpus stays on disk as a memory-mapped array and
is opened lazily *inside* each DataLoader worker. Opening it in the parent breaks
under the `spawn` start method that Windows and macOS use, because a `np.memmap`
does not survive pickling.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from bravien.data.pack import SFTExample, collate_sft, count_packed_sequences
from bravien.data.tokenize import load_token_file
from bravien.tokenizer.tokenizer import IGNORE_INDEX
from bravien.utils.seeding import worker_init_fn


@dataclass
class SplitBounds:
    """Token range for one split of a flat corpus."""

    name: str
    start: int
    end: int

    @property
    def tokens(self) -> int:
        return self.end - self.start


def contiguous_splits(
    total_tokens: int, *, val_fraction: float = 0.005, min_val_tokens: int = 4096
) -> tuple[SplitBounds, SplitBounds]:
    """Split a token stream into train and validation ranges.

    The validation set is the *tail* of the corpus rather than a random sample.
    Random positions inside a concatenated stream overlap the training blocks
    around them, which makes validation loss optimistic for reasons that have
    nothing to do with generalisation (§45).
    """
    if total_tokens < min_val_tokens * 4:
        raise ValueError(
            f"corpus of {total_tokens:,} tokens is too small to hold out "
            f"{min_val_tokens:,} validation tokens; prepare more data"
        )
    val_tokens = max(int(total_tokens * val_fraction), min_val_tokens)
    boundary = total_tokens - val_tokens
    return (
        SplitBounds("train", 0, boundary),
        SplitBounds("validation", boundary, total_tokens),
    )


class PretrainDataset(Dataset):
    """Fixed-length blocks from a memory-mapped token stream.

    Each item is `seq_len + 1` tokens: the model shifts internally, so the extra
    token supplies a target for the final position instead of wasting it.
    """

    def __init__(
        self,
        token_path: str | Path,
        seq_len: int,
        *,
        bounds: SplitBounds | None = None,
    ) -> None:
        self.token_path = Path(token_path)
        self.seq_len = seq_len

        # Read the sidecar now so length and dtype are known without holding the
        # memmap open across process boundaries.
        sidecar = json.loads(
            self.token_path.with_suffix(".json").read_text(encoding="utf-8")
        )
        self.info: dict[str, Any] = sidecar
        self.dtype = np.dtype(sidecar["dtype"])
        total = int(sidecar["tokens"])

        self.bounds = bounds or SplitBounds("all", 0, total)
        if self.bounds.end > total:
            raise ValueError(
                f"split ends at {self.bounds.end:,} but the corpus holds "
                f"{total:,} tokens"
            )

        self._length = count_packed_sequences(self.bounds.tokens, seq_len)
        if self._length == 0:
            raise ValueError(
                f"split {self.bounds.name!r} has {self.bounds.tokens:,} tokens, "
                f"too few for even one sequence of {seq_len}"
            )
        self._tokens: np.memmap | None = None

    def _array(self) -> np.memmap:
        if self._tokens is None:
            self._tokens, _ = load_token_file(self.token_path)
        return self._tokens

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if not 0 <= index < self._length:
            raise IndexError(index)
        start = self.bounds.start + index * self.seq_len
        window = self._array()[start : start + self.seq_len + 1]
        # int64 for the embedding lookup; the on-disk dtype is uint16.
        ids = torch.from_numpy(np.asarray(window, dtype=np.int64))
        return {"input_ids": ids, "labels": ids.clone()}

    @property
    def total_tokens(self) -> int:
        return self._length * self.seq_len

    def describe(self) -> str:
        return (
            f"{self.bounds.name}: {len(self):,} sequences x {self.seq_len} tokens "
            f"= {self.total_tokens:,} supervised tokens "
            f"(from {self.bounds.tokens:,} available)"
        )


class SFTDataset(Dataset):
    """In-memory instruction-tuning examples.

    Held in RAM deliberately: instruction datasets are thousands of examples, not
    billions of tokens, and keeping them in memory makes shuffling trivial.
    """

    def __init__(self, examples: Sequence[SFTExample]) -> None:
        if not examples:
            raise ValueError("SFTDataset requires at least one example")
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> SFTExample:
        return self.examples[index]

    @property
    def supervised_tokens(self) -> int:
        return sum(ex.num_supervised for ex in self.examples)

    def describe(self) -> str:
        total = sum(len(ex) for ex in self.examples)
        supervised = self.supervised_tokens
        share = supervised / total * 100 if total else 0.0
        return (
            f"{len(self):,} conversations, {total:,} tokens, "
            f"{supervised:,} supervised ({share:.1f}%)"
        )


def make_pretrain_dataloader(
    dataset: PretrainDataset,
    *,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    seed: int = 0,
    drop_last: bool = True,
) -> DataLoader:
    """DataLoader for pretraining blocks.

    `drop_last=True` keeps every step the same shape, which matters because a
    single short final batch changes the effective learning rate for that step.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )


def make_sft_dataloader(
    dataset: SFTDataset,
    *,
    batch_size: int,
    pad_id: int,
    shuffle: bool = True,
    num_workers: int = 0,
    seed: int = 0,
) -> DataLoader:
    """DataLoader that pads each batch to its own longest example."""
    generator = torch.Generator()
    generator.manual_seed(seed)

    def collate(batch: list[SFTExample]) -> dict[str, torch.Tensor]:
        return collate_sft(batch, pad_id)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
    )


def batch_to_device(
    batch: dict[str, torch.Tensor], device: torch.device | str
) -> dict[str, torch.Tensor]:
    """Move a batch, honouring non-blocking transfer when pinned."""
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def supervised_token_count(labels: torch.Tensor) -> int:
    """Positions that actually contribute to the loss.

    Used for throughput reporting: tokens/second measured over padded positions
    would overstate real progress.
    """
    return int((labels != IGNORE_INDEX).sum().item())
