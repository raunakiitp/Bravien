"""The Bravien tokenizer: encoding, decoding, and chat encoding."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from tokenizers import Tokenizer

from bravien.tokenizer.special_tokens import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    SPECIAL_TOKENS,
    UNK_ID,
)
from bravien.tokenizer.templates import Segment, format_conversation, render

TOKENIZER_FILE = "tokenizer.json"
METADATA_FILE = "tokenizer_metadata.json"

#: Loss is not computed at these label positions. Matches
#: `bravien.model.model.IGNORE_INDEX`, duplicated here so the tokenizer package
#: does not need to import torch.
IGNORE_INDEX = -100


@dataclass
class ChatEncoding:
    """A tokenised conversation plus its supervision mask."""

    input_ids: list[int]
    labels: list[int]
    text: str

    def __len__(self) -> int:
        return len(self.input_ids)

    @property
    def num_trainable(self) -> int:
        return sum(1 for label in self.labels if label != IGNORE_INDEX)


class BravienTokenizer:
    """Byte-level BPE tokenizer with Bravien's special tokens and template.

    Byte-level means every possible input maps to some token sequence, so no
    text is ever unrepresentable and no language is excluded — `<UNK>` exists
    for format compatibility but is never emitted by encoding.
    """

    def __init__(self, tokenizer: Tokenizer, metadata: dict[str, Any] | None = None):
        self._tok = tokenizer
        self.metadata = metadata or {}
        self._verify_special_tokens()

    def _verify_special_tokens(self) -> None:
        """Fail fast if the reserved ids drifted.

        A tokenizer whose PAD is not 0 will silently poison training, so this
        is checked on every load rather than trusted.
        """
        for expected_id, token in enumerate(SPECIAL_TOKENS):
            actual = self._tok.token_to_id(token)
            if actual is None:
                raise ValueError(
                    f"tokenizer is missing required special token {token!r}"
                )
            if actual != expected_id:
                raise ValueError(
                    f"special token {token!r} has id {actual}, expected "
                    f"{expected_id}; this tokenizer is incompatible with Bravien "
                    f"checkpoints"
                )

    # ------------------------------------------------------------ loading

    @classmethod
    def from_pretrained(cls, path: str | Path) -> BravienTokenizer:
        path = Path(path)
        tok_path = path / TOKENIZER_FILE if path.is_dir() else path
        if not tok_path.exists():
            raise FileNotFoundError(f"no tokenizer at {tok_path}")

        tokenizer = Tokenizer.from_file(str(tok_path))

        metadata: dict[str, Any] = {}
        meta_path = tok_path.parent / METADATA_FILE
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))

        return cls(tokenizer, metadata)

    def save_pretrained(self, path: str | Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._tok.save(str(path / TOKENIZER_FILE))
        (path / METADATA_FILE).write_text(
            json.dumps(self.metadata, indent=2), encoding="utf-8"
        )
        return path

    # ---------------------------------------------------------- properties

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    @property
    def pad_token_id(self) -> int:
        return PAD_ID

    @property
    def bos_token_id(self) -> int:
        return BOS_ID

    @property
    def eos_token_id(self) -> int:
        return EOS_ID

    @property
    def unk_token_id(self) -> int:
        return UNK_ID

    @property
    def backend(self) -> Tokenizer:
        """Escape hatch to the underlying `tokenizers.Tokenizer`."""
        return self._tok

    # ------------------------------------------------------------ encoding

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = self._tok.encode(text, add_special_tokens=False).ids
        if add_special_tokens:
            return [BOS_ID, *ids, EOS_ID]
        return ids

    def encode_batch(
        self, texts: Iterable[str], add_special_tokens: bool = False
    ) -> list[list[int]]:
        encodings = self._tok.encode_batch(
            list(texts), add_special_tokens=False
        )
        if add_special_tokens:
            return [[BOS_ID, *e.ids, EOS_ID] for e in encodings]
        return [e.ids for e in encodings]

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def token_to_id(self, token: str) -> int | None:
        return self._tok.token_to_id(token)

    def id_to_token(self, token_id: int) -> str | None:
        return self._tok.id_to_token(token_id)

    # -------------------------------------------------------- chat encoding

    def encode_segments(self, segments: Iterable[Segment]) -> ChatEncoding:
        """Encode template segments, tracking which tokens are targets.

        Segments are encoded independently and concatenated. That is safe here
        because every content segment is bounded by special tokens, which are
        hard splits for the pre-tokenizer — so no BPE merge can straddle a
        boundary and the result matches encoding the joined string.
        """
        input_ids: list[int] = []
        labels: list[int] = []
        parts: list[str] = []

        for seg in segments:
            ids = self._tok.encode(seg.text, add_special_tokens=False).ids
            input_ids.extend(ids)
            labels.extend(ids if seg.trainable else [IGNORE_INDEX] * len(ids))
            parts.append(seg.text)

        return ChatEncoding(input_ids=input_ids, labels=labels, text="".join(parts))

    def encode_chat(
        self,
        messages: Iterable[dict[str, Any]],
        *,
        add_generation_prompt: bool = False,
        add_bos: bool = True,
        add_eos: bool = False,
    ) -> ChatEncoding:
        segments = format_conversation(
            messages,
            add_generation_prompt=add_generation_prompt,
            add_bos=add_bos,
            add_eos=add_eos,
        )
        return self.encode_segments(segments)

    def apply_chat_template(self, messages: Iterable[dict[str, Any]]) -> str:
        """The literal prompt string sent to the model at inference."""
        return render(format_conversation(messages, add_generation_prompt=True))

    # -------------------------------------------------------- inspection

    def stats(self) -> dict[str, Any]:
        vocab = self._tok.get_vocab()
        return {
            "vocab_size": self.vocab_size,
            "num_special_tokens": len(SPECIAL_TOKENS),
            "special_tokens": {t: self._tok.token_to_id(t) for t in SPECIAL_TOKENS},
            "longest_token": max(vocab, key=len) if vocab else None,
            "metadata": self.metadata,
        }

    def roundtrip_ok(self, text: str) -> bool:
        """Whether `decode(encode(text)) == text` exactly."""
        return self.decode(self.encode(text), skip_special_tokens=False) == text

    def __repr__(self) -> str:
        return (
            f"BravienTokenizer(vocab_size={self.vocab_size}, "
            f"specials={len(SPECIAL_TOKENS)})"
        )
