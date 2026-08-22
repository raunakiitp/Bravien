"""Text normalisation (§12, stage 2).

Cleaning is deliberately conservative: it fixes encoding and whitespace damage
without rewriting meaning. Anything that would change what a document *says*
belongs in `filter.py` as a drop decision, not here as a silent edit.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

# C0/C1 control characters except tab and newline, plus zero-width and
# bidirectional-override characters used to obfuscate text.
_CONTROL_CHARS = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f​-‏‪-‮﻿]"
)
_CARRIAGE_RETURNS = re.compile(r"\r\n?")
_TRAILING_SPACE = re.compile(r"[ \t]+$", re.MULTILINE)
_MANY_BLANK_LINES = re.compile(r"\n{4,}")
_MANY_SPACES = re.compile(r"[ \t]{4,}")
_HTML_TAG = re.compile(r"<[^>\n]{1,200}>")
_HTML_ENTITY = re.compile(r"&(?:#\d{1,6}|#x[0-9a-fA-F]{1,5}|[a-zA-Z]{2,10});")
_URL = re.compile(r"https?://\S+")

_ENTITIES = {
    "&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"',
    "&apos;": "'", "&nbsp;": " ", "&#39;": "'", "&#34;": '"',
}


@dataclass
class Document:
    """One unit of training text as it moves through the pipeline."""

    text: str
    source: str = "unknown"
    doc_id: str = ""
    language: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.text)


@dataclass
class CleanOptions:
    normalize_unicode: bool = True
    unicode_form: str = "NFC"
    strip_control: bool = True
    normalize_newlines: bool = True
    collapse_whitespace: bool = True
    unescape_entities: bool = True
    strip_html: bool = False
    replace_urls_with: str | None = None


def unescape_entities(text: str) -> str:
    for entity, char in _ENTITIES.items():
        if entity in text:
            text = text.replace(entity, char)
    return text


def clean_text(text: str, options: CleanOptions | None = None) -> str:
    """Normalise one document's text."""
    opts = options or CleanOptions()

    if opts.normalize_unicode:
        text = unicodedata.normalize(opts.unicode_form, text)
    if opts.unescape_entities:
        text = unescape_entities(text)
    if opts.strip_html:
        text = _HTML_TAG.sub(" ", text)
    if opts.replace_urls_with is not None:
        text = _URL.sub(opts.replace_urls_with, text)
    if opts.normalize_newlines:
        text = _CARRIAGE_RETURNS.sub("\n", text)
    if opts.strip_control:
        text = _CONTROL_CHARS.sub("", text)
    if opts.collapse_whitespace:
        text = _MANY_SPACES.sub("  ", text)
        text = _TRAILING_SPACE.sub("", text)
        text = _MANY_BLANK_LINES.sub("\n\n\n", text)

    return text.strip()


def clean_document(doc: Document, options: CleanOptions | None = None) -> Document:
    doc.text = clean_text(doc.text, options)
    return doc


def looks_like_html(text: str, threshold: float = 0.002) -> bool:
    """Whether a document carries enough markup to warrant HTML stripping."""
    if not text:
        return False
    tags = len(_HTML_TAG.findall(text)) + len(_HTML_ENTITY.findall(text))
    return tags / len(text) > threshold
