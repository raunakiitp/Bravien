"""Deduplication (§12, stage 4).

Two passes, cheapest first:

* **Exact**: SHA-256 of the normalised text. Catches mirrored pages and repeated
  crawls, and costs one hash per document.
* **Near**: MinHash over character shingles, bucketed by LSH bands. Catches
  boilerplate-wrapped reposts that differ by a header or a date.

Duplicate training data is not merely wasteful — a document seen a hundred times
is effectively memorised, which inflates any evaluation that happens to overlap
it (§45).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from bravien.data.clean import Document

_WHITESPACE = re.compile(r"\s+")

#: 64-bit Mersenne prime, used as the modulus for the hash permutations.
_MERSENNE_61 = (1 << 61) - 1


def normalize_for_hashing(text: str) -> str:
    """Collapse formatting so cosmetic differences do not defeat exact matching."""
    return _WHITESPACE.sub(" ", text.strip().lower())


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize_for_hashing(text).encode("utf-8")).hexdigest()


def shingles(text: str, size: int = 5) -> set[str]:
    """Word-level n-gram shingles.

    Words rather than characters: word shingles are cheaper for long documents
    and are what the standard MinHash/Jaccard estimates assume.
    """
    words = normalize_for_hashing(text).split()
    if len(words) < size:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + size]) for i in range(len(words) - size + 1)}


class MinHasher:
    """MinHash signatures via universal hashing.

    `num_permutations` trades accuracy for cost: 128 estimates Jaccard similarity
    to roughly +/-0.09, which is ample for a near-duplicate threshold of 0.8.
    """

    def __init__(self, num_permutations: int = 128, seed: int = 0) -> None:
        if num_permutations <= 0:
            raise ValueError("num_permutations must be positive")
        self.num_permutations = num_permutations
        # Deterministic coefficients: the same corpus must dedupe identically on
        # every run (§23), so these are derived from the seed, not from random().
        self._a: list[int] = []
        self._b: list[int] = []
        for i in range(num_permutations):
            digest = hashlib.blake2b(
                f"{seed}:{i}".encode(), digest_size=16
            ).digest()
            a = int.from_bytes(digest[:8], "big") % (_MERSENNE_61 - 1) + 1
            b = int.from_bytes(digest[8:], "big") % _MERSENNE_61
            self._a.append(a)
            self._b.append(b)

    def signature(self, features: set[str]) -> tuple[int, ...]:
        if not features:
            return tuple([_MERSENNE_61] * self.num_permutations)

        base = [
            int.from_bytes(
                hashlib.blake2b(f.encode("utf-8"), digest_size=8).digest(), "big"
            )
            % _MERSENNE_61
            for f in features
        ]

        signature = []
        for a, b in zip(self._a, self._b, strict=True):
            signature.append(min((a * h + b) % _MERSENNE_61 for h in base))
        return tuple(signature)


def jaccard_from_signatures(
    left: tuple[int, ...], right: tuple[int, ...]
) -> float:
    """Estimate Jaccard similarity as the fraction of matching hash minima."""
    if len(left) != len(right):
        raise ValueError("signatures have different lengths")
    matches = sum(1 for a, b in zip(left, right, strict=True) if a == b)
    return matches / len(left)


@dataclass
class DedupStats:
    seen: int = 0
    kept: int = 0
    exact_duplicates: int = 0
    near_duplicates: int = 0
    dropped_by_source: dict[str, int] = field(default_factory=dict)

    @property
    def dropped(self) -> dict[str, int]:
        return {
            "exact_duplicate": self.exact_duplicates,
            "near_duplicate": self.near_duplicates,
        }


class Deduplicator:
    """Streaming deduplicator: one pass, bounded memory per document.

    Memory grows with the number of *kept* documents (one hash plus one signature
    each), not with corpus size on disk.
    """

    def __init__(
        self,
        *,
        near_duplicates: bool = True,
        threshold: float = 0.8,
        num_permutations: int = 128,
        bands: int = 16,
        shingle_size: int = 5,
        seed: int = 0,
    ) -> None:
        if not 0 < threshold <= 1:
            raise ValueError("threshold must be in (0, 1]")
        if near_duplicates and num_permutations % bands != 0:
            raise ValueError(
                f"num_permutations ({num_permutations}) must be divisible by "
                f"bands ({bands})"
            )

        self.near_duplicates = near_duplicates
        self.threshold = threshold
        self.bands = bands
        self.rows = num_permutations // bands if near_duplicates else 0
        self.shingle_size = shingle_size
        self.stats = DedupStats()

        self._hasher = MinHasher(num_permutations, seed) if near_duplicates else None
        self._exact: set[str] = set()
        self._signatures: list[tuple[int, ...]] = []
        # band index -> bucket key -> ids of kept documents in that bucket
        self._buckets: list[dict[tuple[int, ...], list[int]]] = [
            {} for _ in range(bands)
        ]

    def _band_keys(self, signature: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
        for band in range(self.bands):
            yield signature[band * self.rows : (band + 1) * self.rows]

    def is_duplicate(self, text: str) -> str | None:
        """Check a document and record it as seen. Returns a reason, or None.

        Registering happens here so callers cannot forget to, which would make
        the very first duplicate pair both pass.
        """
        self.stats.seen += 1

        digest = content_hash(text)
        if digest in self._exact:
            self.stats.exact_duplicates += 1
            return "exact_duplicate"

        if self._hasher is None:
            self._exact.add(digest)
            self.stats.kept += 1
            return None

        signature = self._hasher.signature(shingles(text, self.shingle_size))

        # Only documents sharing a whole band are compared: that is what makes
        # this sub-quadratic.
        candidates: set[int] = set()
        for band, key in enumerate(self._band_keys(signature)):
            candidates.update(self._buckets[band].get(key, ()))

        for candidate in candidates:
            if jaccard_from_signatures(signature, self._signatures[candidate]) >= (
                self.threshold
            ):
                self.stats.near_duplicates += 1
                return "near_duplicate"

        doc_id = len(self._signatures)
        self._signatures.append(signature)
        for band, key in enumerate(self._band_keys(signature)):
            self._buckets[band].setdefault(key, []).append(doc_id)
        self._exact.add(digest)
        self.stats.kept += 1
        return None

    def filter(self, documents: Iterable[Document]) -> Iterator[Document]:
        """Yield only the first occurrence of each distinct document."""
        for doc in documents:
            reason = self.is_duplicate(doc.text)
            if reason is None:
                yield doc
            else:
                key = f"{reason}:{doc.source}"
                self.stats.dropped_by_source[key] = (
                    self.stats.dropped_by_source.get(key, 0) + 1
                )

    def filter_examples(self, examples: Iterable[Any]) -> Iterator[Any]:
        """Yield only the first occurrence of each distinct conversation example."""
        for ex in examples:
            msgs = ex.messages if hasattr(ex, "messages") else ex.get("messages", [])
            parts = []
            for m in msgs:
                r = m.role if hasattr(m, "role") else m.get("role", "")
                c = m.content if hasattr(m, "content") else m.get("content", "")
                parts.append(f"{r}:{c.strip()}")
            text = " ".join(parts)
            reason = self.is_duplicate(text)
            if reason is None:
                yield ex
            else:
                src = getattr(ex, "source", "unknown")
                key = f"{reason}:{src}"
                self.stats.dropped_by_source[key] = (
                    self.stats.dropped_by_source.get(key, 0) + 1
                )


def deduplicate(
    documents: Iterable[Document], **kwargs: object
) -> tuple[list[Document], DedupStats]:
    """Convenience wrapper for corpora small enough to hold in memory."""
    dedup = Deduplicator(**kwargs)  # type: ignore[arg-type]
    return list(dedup.filter(documents)), dedup.stats
