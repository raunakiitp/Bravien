"""Safety policies governing memory retention and secret filtering."""

from __future__ import annotations

import re


class MemoryPolicy:
    """Evaluates whether information is safe and appropriate for persistent memory storage."""

    FORBIDDEN_SECRET_PATTERNS = [
        re.compile(r"\b(password|passwd|pwd)\b", re.IGNORECASE),
        re.compile(r"\b(api[_-]?key|secret[_-]?key|auth[_-]?token)\b", re.IGNORECASE),
        re.compile(r"\b(bearer\s+[a-zA-Z0-9\._\-]{20,})\b", re.IGNORECASE),
        re.compile(r"\b(ghp_[a-zA-Z0-9]{36}|gho_[a-zA-Z0-9]{36})\b"),
        re.compile(r"\b(sk-[a-zA-Z0-9]{32,})\b"),
        re.compile(r"\b(-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----)\b"),
        re.compile(r"\b(\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4})\b"),  # Credit card numbers
    ]

    @classmethod
    def is_safe_to_store(cls, key: str, value: str) -> tuple[bool, str]:
        combined = f"{key} {value}"
        for pat in cls.FORBIDDEN_SECRET_PATTERNS:
            if pat.search(combined):
                return False, "Memory storage rejected: contains sensitive authentication credential, private key, or financial secret."
        return True, "Safe to store."
