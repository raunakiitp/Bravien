"""Efficient Token Packing and Pretraining Dataset Formatter for Bravien."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import Dataset

from bravien.tokenizer.tokenizer import BravienTokenizer


@dataclass
class DatasetCursor:
    """Tracks position in pretraining stream for exact training resumption."""

    sample_index: int = 0
    token_offset: int = 0
    epoch: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "sample_index": self.sample_index,
            "token_offset": self.token_offset,
            "epoch": self.epoch,
        }

    @classmethod
    def from_dict(cls, data: dict[str, int]) -> DatasetCursor:
        return cls(
            sample_index=data.get("sample_index", 0),
            token_offset=data.get("token_offset", 0),
            epoch=data.get("epoch", 0),
        )


class PackedPretrainingDataset(Dataset):
    """Memory-efficient packed pretraining dataset supporting fixed sequence lengths.

    Concatenates variable-length documents into contiguous blocks of `max_seq_len`
    separated by `<EOS>` tokens, maximizing computational efficiency per GPU forward pass.
    """

    def __init__(
        self,
        token_blocks: list[torch.Tensor],
        max_seq_len: int = 2048,
    ) -> None:
        self.blocks = token_blocks
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        tokens = self.blocks[idx]
        return {
            "input_ids": tokens,
            "labels": tokens.clone(),
        }

    @classmethod
    def from_texts(
        cls,
        texts: list[str] | Iterator[str],
        tokenizer: BravienTokenizer,
        max_seq_len: int = 2048,
    ) -> PackedPretrainingDataset:
        """Tokenizes documents and packs them into contiguous blocks of max_seq_len."""
        blocks: list[torch.Tensor] = []
        current_buffer: list[int] = []

        eos_id = tokenizer.eos_token_id
        bos_id = tokenizer.bos_token_id

        for text in texts:
            # Tokenize document without special tokens (we insert manually)
            doc_tokens = tokenizer.encode(text, add_special_tokens=False)
            if not doc_tokens:
                continue

            current_buffer.append(bos_id)
            current_buffer.extend(doc_tokens)
            current_buffer.append(eos_id)

            while len(current_buffer) >= max_seq_len:
                chunk = current_buffer[:max_seq_len]
                blocks.append(torch.tensor(chunk, dtype=torch.long))
                current_buffer = current_buffer[max_seq_len:]

        # Handle remaining tokens
        if len(current_buffer) > 0:
            # Pad to max_seq_len
            pad_id = tokenizer.pad_token_id
            padded = current_buffer + [pad_id] * (max_seq_len - len(current_buffer))
            blocks.append(torch.tensor(padded, dtype=torch.long))

        if not blocks:
            # Create a minimum fallback block
            blocks.append(torch.full((max_seq_len,), tokenizer.pad_token_id, dtype=torch.long))

        return cls(blocks, max_seq_len=max_seq_len)

    def save(self, file_path: str | Path) -> None:
        """Save packed blocks to disk in PyTorch format."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "max_seq_len": self.max_seq_len,
                "num_blocks": len(self.blocks),
                "blocks": self.blocks,
            },
            path,
        )

    @classmethod
    def load(cls, file_path: str | Path) -> PackedPretrainingDataset:
        data = torch.load(file_path, weights_only=True)
        return cls(token_blocks=data["blocks"], max_seq_len=data["max_seq_len"])
