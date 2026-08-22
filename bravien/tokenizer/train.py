"""Tokenizer training (§11).

Byte-level BPE via the `tokenizers` Rust trainer — a real, well-established
training method, run locally. Nothing here calls out to a service.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

from bravien.tokenizer.special_tokens import SPECIAL_TOKENS
from bravien.tokenizer.tokenizer import BravienTokenizer

#: Sentences exercising scripts, punctuation, whitespace and code, used to prove
#: the trained tokenizer round-trips before it is written to disk.
ROUNDTRIP_PROBES: tuple[str, ...] = (
    "Bravien is a language model trained from scratch.",
    "def add(a, b):\n    return a + b  # trailing comment",
    "Unicode: café, naïve, Ω≈ç√, 日本語のテキスト, Привет, مرحبا",
    "Emoji survive byte-level BPE: 🚀🧠",
    "   leading and trailing whitespace   ",
    "Mixed\ttabs\nand\r\nnewlines",
    "Numbers 3.14159 and 1,000,000 and -42",
)


class TokenizerTrainingError(RuntimeError):
    pass


def _counting_iterator(
    texts: Iterable[str], stats: dict[str, int]
) -> Iterator[str]:
    """Pass text through while accumulating corpus statistics.

    Wrapping the iterator keeps training streaming — the corpus is never held
    in memory in full (§11).
    """
    for text in texts:
        if not text:
            continue
        stats["documents"] += 1
        stats["characters"] += len(text)
        stats["bytes"] += len(text.encode("utf-8"))
        yield text


def build_tokenizer(vocab_size: int, min_frequency: int = 2) -> tuple[Tokenizer, Any]:
    """Construct an untrained byte-level BPE tokenizer and its trainer."""
    tokenizer = Tokenizer(models.BPE(unk_token=None))

    # NFC keeps visually identical strings from getting distinct token ids.
    tokenizer.normalizer = normalizers.NFC()

    # Byte-level pre-tokenization: the alphabet is the 256 byte values, so every
    # possible input is representable and <UNK> is unreachable.
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        # Reserved first, so their ids are 0..len(SPECIAL_TOKENS)-1.
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    return tokenizer, trainer


def train_tokenizer(
    texts: Iterable[str],
    *,
    vocab_size: int = 32000,
    min_frequency: int = 2,
    output_dir: str | Path | None = None,
    sources: list[dict[str, Any]] | None = None,
) -> BravienTokenizer:
    """Train a Bravien tokenizer and verify it before returning.

    Args:
        texts: iterable of documents. Consumed once, streamed.
        vocab_size: total target size *including* special tokens.
        output_dir: if given, the tokenizer and its metadata are written here.
        sources: dataset provenance records to embed in metadata (§14).

    Raises:
        TokenizerTrainingError: if the corpus is empty or a round-trip probe
            fails to decode back to its exact input.
    """
    if vocab_size <= len(SPECIAL_TOKENS):
        raise ValueError(
            f"vocab_size ({vocab_size}) must exceed the "
            f"{len(SPECIAL_TOKENS)} reserved special tokens"
        )

    tokenizer, trainer = build_tokenizer(vocab_size, min_frequency)

    stats = {"documents": 0, "characters": 0, "bytes": 0}
    started = time.perf_counter()
    tokenizer.train_from_iterator(_counting_iterator(texts, stats), trainer=trainer)
    elapsed = time.perf_counter() - started

    if stats["documents"] == 0:
        raise TokenizerTrainingError(
            "corpus was empty — nothing to train on. Check the data manifest."
        )

    metadata: dict[str, Any] = {
        "name": "bravien-tokenizer",
        "version": "0.1.0",
        "algorithm": "byte-level BPE",
        "requested_vocab_size": vocab_size,
        "actual_vocab_size": tokenizer.get_vocab_size(),
        "min_frequency": min_frequency,
        "special_tokens": list(SPECIAL_TOKENS),
        "corpus": stats,
        "train_seconds": round(elapsed, 3),
        "sources": sources or [],
        "vocab_checksum": _vocab_checksum(tokenizer),
    }

    bravien_tok = BravienTokenizer(tokenizer, metadata)

    failures = [p for p in ROUNDTRIP_PROBES if not bravien_tok.roundtrip_ok(p)]
    if failures:
        raise TokenizerTrainingError(
            "tokenizer failed round-trip verification on "
            f"{len(failures)} probe(s); first failure: {failures[0]!r}"
        )
    metadata["roundtrip_probes_passed"] = len(ROUNDTRIP_PROBES)

    if output_dir is not None:
        bravien_tok.save_pretrained(output_dir)

    return bravien_tok


def _vocab_checksum(tokenizer: Tokenizer) -> str:
    """Stable hash of the vocabulary, for checkpoint/tokenizer pairing (§23)."""
    vocab = tokenizer.get_vocab()
    payload = json.dumps(
        sorted(vocab.items(), key=lambda kv: kv[1]), ensure_ascii=True
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()[:32]


def summarize(tokenizer: BravienTokenizer, samples: Iterable[str] | None = None) -> str:
    """Human-readable statistics printed after training (§11)."""
    meta = tokenizer.metadata
    corpus = meta.get("corpus", {})
    lines = [
        "Bravien tokenizer",
        "",
        f"  Algorithm:       {meta.get('algorithm', 'byte-level BPE')}",
        f"  Vocab size:      {tokenizer.vocab_size}",
        f"  Special tokens:  {len(SPECIAL_TOKENS)}",
        f"  Checksum:        {meta.get('vocab_checksum', 'n/a')}",
        "",
        "  Corpus",
        f"    documents:     {corpus.get('documents', 0):,}",
        f"    characters:    {corpus.get('characters', 0):,}",
        f"    bytes:         {corpus.get('bytes', 0):,}",
        f"  Trained in:      {meta.get('train_seconds', 0)} s",
    ]

    probes = list(samples) if samples is not None else list(ROUNDTRIP_PROBES[:3])
    if probes:
        lines += ["", "  Encode / decode check"]
        for probe in probes:
            ids = tokenizer.encode(probe)
            ok = tokenizer.roundtrip_ok(probe)
            ratio = len(probe.encode("utf-8")) / max(len(ids), 1)
            preview = probe if len(probe) <= 46 else probe[:43] + "..."
            lines.append(
                f"    {'ok ' if ok else 'FAIL'} {len(ids):4d} tok  "
                f"{ratio:4.2f} bytes/tok  {preview!r}"
            )

    return "\n".join(lines)
