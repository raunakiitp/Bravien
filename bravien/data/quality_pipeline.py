"""High-Performance Pretraining Quality Filter and Dataset Curation Pipeline."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator


@dataclass
class FilterStats:
    """Detailed statistics on filtering, deduplication, and quality metrics."""

    total_documents_scanned: int = 0
    passed_documents: int = 0
    rejected_documents: int = 0
    total_characters_passed: int = 0
    estimated_tokens_passed: int = 0
    duplicates_removed: int = 0
    secrets_blocked: int = 0
    malware_blocked: int = 0
    repetition_rejected: int = 0
    too_short_rejected: int = 0
    contaminated_rejected: int = 0
    domain_distribution: dict[str, int] = field(default_factory=Counter)
    language_distribution: dict[str, int] = field(default_factory=Counter)
    doc_lengths: list[int] = field(default_factory=list)

    @property
    def duplicate_rate_percent(self) -> float:
        if self.total_documents_scanned == 0:
            return 0.0
        return (self.duplicates_removed / self.total_documents_scanned) * 100.0

    @property
    def pass_rate_percent(self) -> float:
        if self.total_documents_scanned == 0:
            return 0.0
        return (self.passed_documents / self.total_documents_scanned) * 100.0

    def compute_percentiles(self) -> dict[str, int]:
        if not self.doc_lengths:
            return {"p50": 0, "p90": 0, "p99": 0, "avg": 0, "max": 0}
        s = sorted(self.doc_lengths)
        n = len(s)
        return {
            "p50": s[int(n * 0.5)],
            "p90": s[int(n * 0.9)],
            "p99": s[int(n * 0.99)],
            "avg": int(sum(s) / n),
            "max": s[-1],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_documents_scanned": self.total_documents_scanned,
            "passed_documents": self.passed_documents,
            "rejected_documents": self.rejected_documents,
            "pass_rate_percent": round(self.pass_rate_percent, 2),
            "duplicate_rate_percent": round(self.duplicate_rate_percent, 2),
            "total_characters_passed": self.total_characters_passed,
            "estimated_tokens_passed": self.estimated_tokens_passed,
            "reasons": {
                "duplicates": self.duplicates_removed,
                "secrets": self.secrets_blocked,
                "malware": self.malware_blocked,
                "repetition": self.repetition_rejected,
                "too_short": self.too_short_rejected,
                "contaminated": self.contaminated_rejected,
            },
            "domain_distribution": dict(self.domain_distribution),
            "language_distribution": dict(self.language_distribution),
            "length_metrics": self.compute_percentiles(),
        }


class QualityFilter:
    """Document filter enforcing safety, deduplication, formatting, and high-density information."""

    # Secret scanning regexes
    RE_SECRET_PATTERNS = [
        re.compile(r"(?:AKIA|ABIA|ACCA|ASIA)[0-9A-Z]{16}"),  # AWS Access Key
        re.compile(r"gh[pousr]_[0-9a-zA-Z]{36}"),             # GitHub Personal Token
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"), # Private Keys
        re.compile(r"(?:api[_-]?key|secret[_-]?key|auth[_-]?token)[\s:=]+['\"][a-zA-Z0-9_\-]{20,}['\"]", re.IGNORECASE),
    ]

    # Dangerous payload filtering
    RE_MALWARE_PATTERNS = [
        re.compile(r"\b(?:eval\s*\(\s*base64_decode|powershell\s+-enc|cmd\.exe\s+/c\s+echo|rm\s+-rf\s+/\b)"),
        re.compile(r"\b(?:mimikatz|msfvenom|meterpreter|reverse_tcp|bind_shell)\b", re.IGNORECASE),
    ]

    # Benchmark contamination check (protect test set leakage)
    CONTAMINATION_SUBSTRINGS = [
        "What will be the exact price of gold on December 31, 2035",
        "DEV_OVERRIDE_ENABLED",
        "Ignore all prior developer constraints",
        "A factory has 3 machines that produce 150 widgets in 5 hours",
    ]

    def __init__(self, min_char_len: int = 50, max_char_len: int = 100_000) -> None:
        self.min_char_len = min_char_len
        self.max_char_len = max_char_len
        self.seen_hashes: set[str] = set()
        self.stats = FilterStats()

    def _detect_language(self, text: str) -> str:
        # Check Devanagari range for Hindi
        devanagari_count = sum(1 for c in text if "\u0900" <= c <= "\u097F")
        if devanagari_count > 10:
            return "hi"
        # Check Hinglish keywords
        hinglish_words = ["hai", "karna", "hoga", "kya", "kaise", "batao", "mujhe", "aapka", "karo"]
        words = text.lower().split()
        if any(w in words for w in hinglish_words):
            return "hinglish"
        # Check code indicators
        code_markers = ["def ", "class ", "function ", "import ", "const ", "return ", "interface "]
        if sum(1 for m in code_markers if m in text) >= 2:
            return "code"
        return "en"

    def _has_excessive_repetition(self, text: str) -> bool:
        # Check single-character repetition (e.g. "aaaaaa...")
        if re.search(r"(.)\1{9,}", text):
            return True
        # Check line repetition
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if len(lines) >= 6:
            line_counts = Counter(lines)
            most_common, count = line_counts.most_common(1)[0]
            if count / len(lines) > 0.4:
                return True
        return False

    def validate_and_filter(self, text: str, domain: str = "general_web") -> tuple[bool, str, str]:
        """Validate document. Returns (is_valid, reason, language)."""
        self.stats.total_documents_scanned += 1

        # 1. Unicode Normalization
        text = unicodedata.normalize("NFC", text.strip())

        # 2. Length check
        if len(text) < self.min_char_len:
            self.stats.rejected_documents += 1
            self.stats.too_short_rejected += 1
            return False, "too_short", "unknown"

        if len(text) > self.max_char_len:
            text = text[: self.max_char_len]

        # 3. Secret scanning
        for pattern in self.RE_SECRET_PATTERNS:
            if pattern.search(text):
                self.stats.rejected_documents += 1
                self.stats.secrets_blocked += 1
                return False, "secret_detected", "unknown"

        # 4. Malware payload filter
        for pattern in self.RE_MALWARE_PATTERNS:
            if pattern.search(text):
                self.stats.rejected_documents += 1
                self.stats.malware_blocked += 1
                return False, "malware_flagged", "unknown"

        # 5. Contamination filter
        for probe in self.CONTAMINATION_SUBSTRINGS:
            if probe.lower() in text.lower():
                self.stats.rejected_documents += 1
                self.stats.contaminated_rejected += 1
                return False, "benchmark_contaminated", "unknown"

        # 6. Repetition check
        if self._has_excessive_repetition(text):
            self.stats.rejected_documents += 1
            self.stats.repetition_rejected += 1
            return False, "excessive_repetition", "unknown"

        # 7. Exact deduplication (SHA-256)
        doc_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if doc_hash in self.seen_hashes:
            self.stats.rejected_documents += 1
            self.stats.duplicates_removed += 1
            return False, "duplicate", "unknown"

        self.seen_hashes.add(doc_hash)

        # Document Passed
        lang = self._detect_language(text)
        self.stats.passed_documents += 1
        self.stats.total_characters_passed += len(text)
        self.stats.estimated_tokens_passed += max(1, int(len(text) / 4.0))
        self.stats.domain_distribution[domain] += 1
        self.stats.language_distribution[lang] += 1
        self.stats.doc_lengths.append(len(text))

        return True, "passed", lang


# Semantic alias
DocumentQualityFilter = QualityFilter
