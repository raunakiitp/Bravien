"""Quality, language, and PII filtering (§12, §13).

Filters return a *reason* when they drop a document, and those reasons are
aggregated into the manifest. That is what makes "the dataset is clean" an
auditable claim rather than an assertion (§13).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from bravien.data.clean import Document

# --------------------------------------------------------------- language

# Unicode ranges that identify a script unambiguously. Latin is handled
# separately because many languages share it.
_SCRIPT_RANGES: list[tuple[str, int, int]] = [
    ("cyrillic", 0x0400, 0x04FF),
    ("greek", 0x0370, 0x03FF),
    ("hebrew", 0x0590, 0x05FF),
    ("arabic", 0x0600, 0x06FF),
    ("devanagari", 0x0900, 0x097F),
    ("bengali", 0x0980, 0x09FF),
    ("thai", 0x0E00, 0x0E7F),
    ("hangul", 0xAC00, 0xD7AF),
    ("hiragana", 0x3040, 0x309F),
    ("katakana", 0x30A0, 0x30FF),
    ("han", 0x4E00, 0x9FFF),
]

_ENGLISH_STOPWORDS = frozenset(
    """the be to of and a in that have i it for not on with he as you do at this
    but his by from they we say her she or an will my one all would there their
    what so up out if about who get which go me when make can like time no just
    him know take people into year your good some could them see other than then
    now look only come its over think also back after use two how our work first
    well way even new want because any these give day most us is are was were""".split()
)

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def detect_language(text: str, sample_chars: int = 4000) -> str:
    """Heuristic language identification.

    Returns an ISO-639-1-style code, a script name, or "unknown".

    This is a *heuristic*, not a trained classifier: it identifies non-Latin
    scripts reliably, distinguishes English by stopword density, and labels
    everything else "latin-other". It is not a substitute for fastText/CLD3 and
    is documented as such rather than presented as language detection (§13).
    """
    if not text.strip():
        return "unknown"

    sample = text[:sample_chars]
    counts: Counter[str] = Counter()
    latin = 0
    for ch in sample:
        cp = ord(ch)
        if not ch.isalpha():
            continue
        if cp < 0x0250:
            latin += 1
            continue
        for name, lo, hi in _SCRIPT_RANGES:
            if lo <= cp <= hi:
                counts[name] += 1
                break

    total_alpha = latin + sum(counts.values())
    if total_alpha == 0:
        return "unknown"

    if counts:
        script, n = counts.most_common(1)[0]
        if n / total_alpha > 0.3:
            # Japanese mixes Han with kana; that mixture is the giveaway.
            if script in ("hiragana", "katakana"):
                return "ja"
            if script == "han":
                kana = counts["hiragana"] + counts["katakana"]
                return "ja" if kana > 0 else "zh"
            return {"hangul": "ko", "cyrillic": "ru", "arabic": "ar",
                    "hebrew": "he", "greek": "el", "devanagari": "hi",
                    "bengali": "bn", "thai": "th"}.get(script, script)

    words = [w.lower() for w in _WORD.findall(sample)]
    if not words:
        return "unknown"
    hits = sum(1 for w in words if w in _ENGLISH_STOPWORDS)
    return "en" if hits / len(words) > 0.08 else "latin-other"


# -------------------------------------------------------------------- PII

_PII_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]{2,}\b"), "<EMAIL>"),
    ("ipv4", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "<IP>"),
    (
        "phone",
        re.compile(r"(?<!\w)(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)[\s.-]?)?\d{3,4}[\s.-]\d{3,4}(?:[\s.-]\d{3,4})?(?!\w)"),
        "<PHONE>",
    ),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "<SSN>"),
    ("card", re.compile(r"\b(?:\d[ -]?){13,19}\b"), "<CARD>"),
]


def _luhn_valid(digits: str) -> bool:
    """Luhn checksum, so ordinary long numbers are not mistaken for cards."""
    nums = [int(c) for c in digits if c.isdigit()]
    if not 13 <= len(nums) <= 19:
        return False
    total = 0
    for i, n in enumerate(reversed(nums)):
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def redact_pii(text: str) -> tuple[str, dict[str, int]]:
    """Replace personal identifiers with placeholders.

    Redaction rather than dropping: a document mentioning one email address is
    still useful training text once the address is gone.
    """
    found: dict[str, int] = {}
    for name, pattern, placeholder in _PII_PATTERNS:
        if name == "card":
            def _sub_card(m: re.Match[str]) -> str:
                if _luhn_valid(m.group(0)):
                    found["card"] = found.get("card", 0) + 1
                    return placeholder
                return m.group(0)

            text = pattern.sub(_sub_card, text)
            continue

        text, n = pattern.subn(placeholder, text)
        if n:
            found[name] = n
    return text, found


# ---------------------------------------------------------------- quality


@dataclass
class QualityThresholds:
    """Gopher-style document heuristics. Every bound is explicit and tunable."""

    min_chars: int = 200
    max_chars: int = 1_000_000
    min_words: int = 40
    max_words: int = 200_000
    min_mean_word_length: float = 2.5
    max_mean_word_length: float = 12.0
    max_symbol_word_ratio: float = 0.15
    min_alpha_word_ratio: float = 0.6
    max_ellipsis_line_ratio: float = 0.3
    max_duplicate_line_ratio: float = 0.3
    max_top_ngram_ratio: float = 0.20
    ngram_size: int = 5
    require_stopwords: bool = True
    min_stopword_hits: int = 2
    allowed_languages: tuple[str, ...] | None = ("en",)


@dataclass
class FilterResult:
    keep: bool
    reason: str = ""


@dataclass
class FilterStats:
    seen: int = 0
    kept: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    pii_redactions: dict[str, int] = field(default_factory=dict)
    languages: dict[str, int] = field(default_factory=dict)

    def record_drop(self, reason: str) -> None:
        self.dropped[reason] = self.dropped.get(reason, 0) + 1

    def record_pii(self, found: dict[str, int]) -> None:
        for k, v in found.items():
            self.pii_redactions[k] = self.pii_redactions.get(k, 0) + v

    def record_language(self, lang: str) -> None:
        self.languages[lang] = self.languages.get(lang, 0) + 1


def repetition_ratio(words: list[str], n: int) -> float:
    """Share of the document occupied by its single most frequent n-gram.

    Catches the degenerate scraped page that repeats one phrase hundreds of
    times, which otherwise passes every length and character check (§13).
    """
    if len(words) < n * 2:
        return 0.0
    grams = Counter(
        tuple(words[i : i + n]) for i in range(len(words) - n + 1)
    )
    _, top = grams.most_common(1)[0]
    return (top * n) / len(words)


def check_quality(
    text: str, thresholds: QualityThresholds | None = None
) -> FilterResult:
    """Apply document-level quality heuristics."""
    t = thresholds or QualityThresholds()

    if len(text) < t.min_chars:
        return FilterResult(False, "too_short_chars")
    if len(text) > t.max_chars:
        return FilterResult(False, "too_long_chars")

    words = text.split()
    if len(words) < t.min_words:
        return FilterResult(False, "too_short_words")
    if len(words) > t.max_words:
        return FilterResult(False, "too_long_words")

    mean_len = sum(len(w) for w in words) / len(words)
    if mean_len < t.min_mean_word_length:
        return FilterResult(False, "mean_word_too_short")
    if mean_len > t.max_mean_word_length:
        return FilterResult(False, "mean_word_too_long")

    alpha_words = sum(1 for w in words if any(c.isalpha() for c in w))
    if alpha_words / len(words) < t.min_alpha_word_ratio:
        return FilterResult(False, "not_enough_alpha_words")

    symbols = sum(text.count(c) for c in "#<>{}[]|\\^~")
    if symbols / len(words) > t.max_symbol_word_ratio:
        return FilterResult(False, "symbol_heavy")

    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if lines:
        ellipsis = sum(1 for ln in lines if ln.endswith(("...", "…")))
        if ellipsis / len(lines) > t.max_ellipsis_line_ratio:
            return FilterResult(False, "ellipsis_boilerplate")
        counts = Counter(lines)
        duplicated = sum(c for c in counts.values() if c > 1)
        if duplicated / len(lines) > t.max_duplicate_line_ratio:
            return FilterResult(False, "duplicate_lines")

    lowered = [w.lower() for w in words]
    if t.require_stopwords:
        hits = sum(1 for w in lowered if w in _ENGLISH_STOPWORDS)
        lang_is_english = t.allowed_languages is None or "en" in t.allowed_languages
        if lang_is_english and hits < t.min_stopword_hits:
            return FilterResult(False, "no_stopwords")

    if repetition_ratio(lowered, t.ngram_size) > t.max_top_ngram_ratio:
        return FilterResult(False, "excessive_repetition")

    return FilterResult(True)


def filter_document(
    doc: Document,
    thresholds: QualityThresholds | None = None,
    stats: FilterStats | None = None,
    *,
    redact: bool = True,
) -> Document | None:
    """Run language, quality, and PII stages over one document.

    Returns the (possibly redacted) document, or None if it was dropped.
    """
    t = thresholds or QualityThresholds()
    st = stats or FilterStats()
    st.seen += 1

    if not doc.text or not doc.text.strip():
        st.record_drop("empty")
        return None

    lang = doc.language or detect_language(doc.text)
    doc.language = lang
    st.record_language(lang)

    if t.allowed_languages is not None and lang not in t.allowed_languages:
        st.record_drop(f"language_{lang}")
        return None

    quality = check_quality(doc.text, t)
    if not quality.keep:
        st.record_drop(quality.reason)
        return None

    if redact:
        doc.text, found = redact_pii(doc.text)
        if found:
            st.record_pii(found)
            doc.meta["pii_redacted"] = found

    st.kept += 1
    return doc
