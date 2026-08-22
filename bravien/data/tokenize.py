"""Tokenizing a corpus to disk (§12, stage 5).

Documents become a single flat array of token ids in a `.bin` file, plus a JSON
sidecar. Two consequences that matter:

* Training reads the file with `numpy.memmap`, so corpus size is bounded by disk
  rather than RAM.
* The token stream is written once and reused across runs, so epochs after the
  first cost no tokenization time.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from bravien.data.clean import Document
from bravien.data.manifest import file_checksum

if TYPE_CHECKING:
    from bravien.tokenizer.tokenizer import BravienTokenizer

#: Format version, so a stale `.bin` from an older layout is rejected loudly.
FORMAT_VERSION = 1


def dtype_for_vocab(vocab_size: int) -> np.dtype:
    """Narrowest integer type that can hold every id.

    uint16 halves both disk footprint and page-cache pressure for the usual
    32k vocabulary, which is the difference between a corpus that fits in RAM
    cache and one that does not.
    """
    if vocab_size <= np.iinfo(np.uint16).max + 1:
        return np.dtype(np.uint16)
    if vocab_size <= np.iinfo(np.uint32).max + 1:
        return np.dtype(np.uint32)
    raise ValueError(f"vocab_size {vocab_size} is implausibly large")


@dataclass
class TokenizeStats:
    documents: int = 0
    characters: int = 0
    tokens: int = 0
    empty_documents: int = 0
    truncated_documents: int = 0
    per_source: dict[str, int] = field(default_factory=dict)

    @property
    def chars_per_token(self) -> float:
        return self.characters / self.tokens if self.tokens else 0.0


class TokenShardWriter:
    """Append token ids to a flat binary file.

    Buffered in blocks so a corpus of millions of short documents does not incur
    a syscall per document.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        vocab_size: int,
        buffer_tokens: int = 1 << 20,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.dtype = dtype_for_vocab(vocab_size)
        self.vocab_size = vocab_size
        self.total = 0
        self._buffer_tokens = buffer_tokens
        self._buffer: list[int] = []
        self._fh = open(self.path, "wb")

    def write(self, ids: Iterable[int]) -> None:
        self._buffer.extend(ids)
        if len(self._buffer) >= self._buffer_tokens:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        array = np.asarray(self._buffer, dtype=np.int64)
        # An out-of-range id would wrap silently under uint16 and poison training
        # with a token the model never learns.
        if array.size and (array.min() < 0 or array.max() >= self.vocab_size):
            raise ValueError(
                f"token id out of range for vocab_size={self.vocab_size}: "
                f"min={int(array.min())} max={int(array.max())}"
            )
        array.astype(self.dtype).tofile(self._fh)
        self.total += array.size
        self._buffer.clear()

    def close(self) -> None:
        self.flush()
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> TokenShardWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def tokenize_documents(
    documents: Iterable[Document],
    tokenizer: BravienTokenizer,
    *,
    append_eos: bool = True,
    max_tokens_per_document: int | None = None,
    stats: TokenizeStats | None = None,
) -> Iterator[list[int]]:
    """Yield one token id list per document.

    Args:
        append_eos: put EOS after each document. This is what teaches the model
            where a document ends; without it, concatenated documents look like
            one endless text and the model never learns to stop.
        max_tokens_per_document: hard cap, counted after tokenizing.
    """
    st = stats if stats is not None else TokenizeStats()

    for doc in documents:
        text = doc.text
        if not text or not text.strip():
            st.empty_documents += 1
            continue

        ids = tokenizer.encode(text, add_special_tokens=False)
        if max_tokens_per_document is not None and len(ids) > max_tokens_per_document:
            ids = ids[:max_tokens_per_document]
            st.truncated_documents += 1
        if append_eos:
            ids.append(tokenizer.eos_token_id)

        st.documents += 1
        st.characters += len(text)
        st.tokens += len(ids)
        st.per_source[doc.source] = st.per_source.get(doc.source, 0) + len(ids)
        yield ids


def tokenize_to_file(
    documents: Iterable[Document],
    tokenizer: BravienTokenizer,
    output_path: str | Path,
    *,
    append_eos: bool = True,
    max_tokens_per_document: int | None = None,
) -> dict[str, Any]:
    """Tokenize a corpus into `output_path` and write its `.json` sidecar.

    Returns the sidecar contents.
    """
    output_path = Path(output_path)
    stats = TokenizeStats()

    writer = TokenShardWriter(output_path, vocab_size=tokenizer.vocab_size)
    try:
        for ids in tokenize_documents(
            documents,
            tokenizer,
            append_eos=append_eos,
            max_tokens_per_document=max_tokens_per_document,
            stats=stats,
        ):
            writer.write(ids)
    finally:
        # close() flushes; reading writer.total before this would miss whatever
        # is still buffered — which for a corpus smaller than the buffer is
        # everything.
        writer.close()

    total_tokens = writer.total
    dtype_name = writer.dtype.name

    if total_tokens == 0:
        raise ValueError(
            f"tokenizing produced 0 tokens into {output_path}: "
            f"{stats.documents} document(s) survived filtering "
            f"({stats.empty_documents} were empty). Check the filter drop "
            f"reasons in the manifest."
        )

    info: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "path": output_path.name,
        "dtype": dtype_name,
        "tokens": total_tokens,
        "documents": stats.documents,
        "characters": stats.characters,
        "chars_per_token": round(stats.chars_per_token, 3),
        "empty_documents": stats.empty_documents,
        "truncated_documents": stats.truncated_documents,
        "tokens_per_source": stats.per_source,
        "vocab_size": tokenizer.vocab_size,
        "tokenizer_checksum": tokenizer.metadata.get("vocab_checksum"),
        "eos_appended": append_eos,
        "checksum": file_checksum(output_path),
    }
    sidecar = output_path.with_suffix(".json")
    sidecar.write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return info


def load_token_file(path: str | Path) -> tuple[np.memmap, dict[str, Any]]:
    """Memory-map a tokenized corpus and return it with its sidecar.

    Raises:
        FileNotFoundError: if the `.bin` or its `.json` sidecar is missing.
        ValueError: on a format version mismatch or a size that disagrees with
            the recorded token count.
    """
    path = Path(path)
    sidecar_path = path.with_suffix(".json")
    if not path.exists():
        raise FileNotFoundError(f"tokenized corpus not found: {path}")
    if not sidecar_path.exists():
        raise FileNotFoundError(
            f"sidecar {sidecar_path.name} is missing; {path.name} cannot be read "
            f"without its dtype and token count"
        )

    info = json.loads(sidecar_path.read_text(encoding="utf-8"))
    version = info.get("format_version")
    if version != FORMAT_VERSION:
        raise ValueError(
            f"{path.name} uses token format v{version}, this build expects "
            f"v{FORMAT_VERSION}; re-run data preparation"
        )

    dtype = np.dtype(info["dtype"])
    expected = int(info["tokens"])
    actual = path.stat().st_size // dtype.itemsize
    if actual != expected:
        raise ValueError(
            f"{path.name} holds {actual:,} tokens but its sidecar claims "
            f"{expected:,} — the file is truncated or was overwritten"
        )

    return np.memmap(path, dtype=dtype, mode="r"), info
