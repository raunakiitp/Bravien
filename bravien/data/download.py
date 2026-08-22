"""Dataset acquisition (§12 stage 1, §14).

Two paths, both first-class:

* **Download** from sources with an explicitly recorded license. Every source in
  the registry carries its license and a URL for it; nothing is fetched without
  one (§14).
* **Offline.** `generate_seed_corpus()` synthesises a small structured corpus
  locally, so the tokenizer, trainer and smoke test all run with no network at
  all (§2). It is labelled as synthetic everywhere it appears, because a model
  trained on it is a plumbing test and not a useful language model (§71).
"""

from __future__ import annotations

import json
import random
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from bravien.data.clean import Document
from bravien.data.manifest import SourceRecord, file_checksum
from bravien.utils.logging import get_logger
from bravien.utils.paths import sanitize_filename

logger = get_logger("data.download")

#: Refuse a response larger than this. A misconfigured URL should not fill the
#: disk (§53).
DEFAULT_MAX_BYTES = 512 * 1024 * 1024
DEFAULT_TIMEOUT = 30.0
_USER_AGENT = "Bravien-DataPrep/0.1 (local training pipeline)"


class DownloadError(RuntimeError):
    pass


@dataclass(frozen=True)
class SourceSpec:
    """A downloadable corpus with its licensing recorded up front."""

    name: str
    url: str
    license: str
    license_url: str
    format: str = "text"
    notes: str = ""
    sha256: str | None = None

    def filename(self) -> str:
        tail = self.url.rsplit("/", 1)[-1] or f"{self.name}.txt"
        return sanitize_filename(f"{self.name}-{tail}")


#: Sources whose licenses permit training use. Kept short and verifiable on
#: purpose: an unchecked URL list is how unlicensed data enters a corpus.
SOURCES: dict[str, SourceSpec] = {
    "gutenberg-alice": SourceSpec(
        name="gutenberg-alice",
        url="https://www.gutenberg.org/files/11/11-0.txt",
        license="Public domain (US) - Project Gutenberg License",
        license_url="https://www.gutenberg.org/policy/license.html",
        notes="Alice's Adventures in Wonderland, Lewis Carroll. Header/footer "
        "boilerplate is stripped by strip_gutenberg_boilerplate().",
    ),
    "gutenberg-sherlock": SourceSpec(
        name="gutenberg-sherlock",
        url="https://www.gutenberg.org/files/1661/1661-0.txt",
        license="Public domain (US) - Project Gutenberg License",
        license_url="https://www.gutenberg.org/policy/license.html",
        notes="The Adventures of Sherlock Holmes, Arthur Conan Doyle.",
    ),
    "gutenberg-frankenstein": SourceSpec(
        name="gutenberg-frankenstein",
        url="https://www.gutenberg.org/files/84/84-0.txt",
        license="Public domain (US) - Project Gutenberg License",
        license_url="https://www.gutenberg.org/policy/license.html",
        notes="Frankenstein, Mary Shelley.",
    ),
    "gutenberg-pride": SourceSpec(
        name="gutenberg-pride",
        url="https://www.gutenberg.org/files/1342/1342-0.txt",
        license="Public domain (US) - Project Gutenberg License",
        license_url="https://www.gutenberg.org/policy/license.html",
        notes="Pride and Prejudice, Jane Austen.",
    ),
}


def download_file(
    url: str,
    destination: str | Path,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    timeout: float = DEFAULT_TIMEOUT,
    overwrite: bool = False,
) -> Path:
    """Stream a URL to disk with a hard size cap and a timeout.

    Written to a `.part` file and renamed on success, so an interrupted download
    can never be mistaken for a complete one.

    Raises:
        DownloadError: on any network failure, or if the response exceeds
            `max_bytes`.
    """
    destination = Path(destination)
    if destination.exists() and not overwrite:
        logger.info("using cached %s", destination.name)
        return destination

    if not url.lower().startswith(("http://", "https://")):
        raise DownloadError(f"refusing non-HTTP url: {url!r}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")

    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    written = 0
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            declared = response.headers.get("Content-Length")
            if declared and int(declared) > max_bytes:
                raise DownloadError(
                    f"{url} declares {int(declared):,} bytes, over the "
                    f"{max_bytes:,} byte limit"
                )
            with open(partial, "wb") as fh:
                while chunk := response.read(1 << 16):
                    written += len(chunk)
                    if written > max_bytes:
                        raise DownloadError(
                            f"{url} exceeded the {max_bytes:,} byte limit"
                        )
                    fh.write(chunk)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
        partial.unlink(missing_ok=True)
        raise DownloadError(f"could not download {url}: {exc}") from exc
    except DownloadError:
        partial.unlink(missing_ok=True)
        raise

    partial.replace(destination)
    logger.info("downloaded %s (%s bytes)", destination.name, f"{written:,}")
    return destination


_GUTENBERG_START = "*** START OF THE PROJECT GUTENBERG EBOOK"
_GUTENBERG_END = "*** END OF THE PROJECT GUTENBERG EBOOK"


def strip_gutenberg_boilerplate(text: str) -> str:
    """Remove the license header and footer, keeping the work itself.

    Falls back to returning the text unchanged if the markers are absent, since a
    changed marker format should not silently discard a whole book.
    """
    upper = text.upper()
    start = upper.find(_GUTENBERG_START)
    if start != -1:
        newline = text.find("\n", start)
        if newline != -1:
            text = text[newline + 1 :]
            upper = text.upper()
    end = upper.find(_GUTENBERG_END)
    if end != -1:
        text = text[:end]
    return text.strip()


def fetch_source(
    spec: SourceSpec,
    cache_dir: str | Path = "datasets/raw",
    *,
    overwrite: bool = False,
) -> tuple[Path, SourceRecord]:
    """Download one registry source and return its file and provenance record."""
    path = download_file(
        spec.url, Path(cache_dir) / spec.filename(), overwrite=overwrite
    )
    record = SourceRecord(
        name=spec.name,
        url=spec.url,
        license=f"{spec.license} <{spec.license_url}>",
        documents=1,
        bytes=path.stat().st_size,
        notes=spec.notes,
        processing=["download"],
    )
    return path, record


def load_text_file(
    path: str | Path,
    *,
    source: str = "",
    strip_gutenberg: bool = True,
    split_paragraphs: bool = True,
    min_chars: int = 200,
) -> Iterator[Document]:
    """Read a plain-text file as one or more documents.

    Splitting on blank lines is what turns a single book into thousands of
    training documents, which is what deduplication and quality filtering
    operate on.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    if strip_gutenberg and _GUTENBERG_START in text.upper():
        text = strip_gutenberg_boilerplate(text)

    name = source or path.stem
    if not split_paragraphs:
        yield Document(text=text, source=name, doc_id=f"{name}:0")
        return

    chunk: list[str] = []
    index = 0
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        chunk.append(block)
        if sum(len(c) for c in chunk) >= min_chars:
            yield Document(
                text="\n\n".join(chunk), source=name, doc_id=f"{name}:{index}"
            )
            index += 1
            chunk = []
    if chunk:
        yield Document(text="\n\n".join(chunk), source=name, doc_id=f"{name}:{index}")


def load_jsonl(
    path: str | Path, *, text_field: str = "text", source: str = ""
) -> Iterator[Document]:
    """Read a JSONL corpus, skipping malformed lines with a warning."""
    path = Path(path)
    name = source or path.stem
    skipped = 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for index, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            text = record.get(text_field)
            if not isinstance(text, str) or not text.strip():
                skipped += 1
                continue
            yield Document(
                text=text,
                source=record.get("source", name),
                doc_id=str(record.get("id", f"{name}:{index}")),
                meta={k: v for k, v in record.items() if k != text_field},
            )
    if skipped:
        logger.warning("skipped %d unusable line(s) in %s", skipped, path.name)


def load_directory(
    directory: str | Path, *, patterns: tuple[str, ...] = ("*.txt", "*.jsonl")
) -> Iterator[Document]:
    """Read every matching file under a directory as documents.

    This is the "use my own data" path: drop files in and prepare them, no
    network involved.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"not a directory: {directory}")
    for pattern in patterns:
        for path in sorted(directory.rglob(pattern)):
            if path.suffix == ".jsonl":
                yield from load_jsonl(path)
            else:
                yield from load_text_file(path)


# ------------------------------------------------------------ offline fallback

_SUBJECTS = (
    "the engineer", "a student", "the librarian", "my neighbour", "the pilot",
    "a gardener", "the physicist", "an architect", "the sailor", "a baker",
    "the cartographer", "a translator", "the astronomer", "a locksmith",
)
_VERBS = (
    "measured", "described", "repaired", "questioned", "sketched", "catalogued",
    "explained", "adjusted", "compared", "assembled", "reviewed", "predicted",
)
_OBJECTS = (
    "the tide charts", "a folding map", "the brass instrument", "two old letters",
    "the harbour lights", "a wooden model", "the printing press", "several drawings",
    "the weather log", "a small telescope", "the copper kettle", "three notebooks",
)
_PLACES = (
    "in the observatory", "beside the canal", "under the pier", "at the workshop",
    "near the old bridge", "inside the greenhouse", "across the courtyard",
    "along the coast road", "behind the market", "within the archive",
)
_CONNECTORS = (
    "because", "although", "after", "while", "before", "since", "whenever",
)
_TOPICS = (
    "tides", "maps", "clocks", "bridges", "gardens", "letters", "harbours",
    "instruments", "notebooks", "telescopes", "presses", "archives",
)


def generate_seed_corpus(
    num_documents: int = 2000, *, seed: int = 0, paragraphs: int = 3
) -> Iterator[Document]:
    """Synthesise a small structured corpus, for use with no network.

    The text is template-generated with a fixed vocabulary, so it has genuine
    statistical structure — consistent syntax, a bounded lexicon, recurring
    collocations — which is exactly what a tiny model needs in order to show a
    falling loss curve within minutes.

    It is *not* natural language and a model trained on it will only produce
    sentences of this shape. Every document is tagged `bravien-seed-synthetic`
    so it can never be mistaken for real corpus data (§71).
    """
    rng = random.Random(seed)
    for index in range(num_documents):
        blocks = []
        for _ in range(paragraphs):
            sentences = []
            for _ in range(rng.randint(3, 6)):
                subject = rng.choice(_SUBJECTS)
                verb = rng.choice(_VERBS)
                obj = rng.choice(_OBJECTS)
                place = rng.choice(_PLACES)
                if rng.random() < 0.45:
                    connector = rng.choice(_CONNECTORS)
                    tail = (
                        f", {connector} {rng.choice(_SUBJECTS)} "
                        f"{rng.choice(_VERBS)} {rng.choice(_OBJECTS)}"
                    )
                else:
                    tail = ""
                sentences.append(
                    f"{subject.capitalize()} {verb} {obj} {place}{tail}."
                )
                # Numeric and punctuation variety, so the tokenizer sees digits
                # and units rather than words alone.
                if rng.random() < 0.3:
                    sentences.append(
                        f"The reading was {rng.randint(1, 99)}"
                        f"{rng.choice(('.2', '.5', '.75', ''))} "
                        f"{rng.choice(('metres', 'degrees', 'minutes', 'grams'))} "
                        f"on day {rng.randint(1, 365)}."
                    )
            blocks.append(" ".join(sentences))

        topic = rng.choice(_TOPICS)
        text = f"Notes on {topic}.\n\n" + "\n\n".join(blocks)
        yield Document(
            text=text,
            source="bravien-seed-synthetic",
            doc_id=f"seed:{index}",
            language="en",
            meta={"synthetic": True, "topic": topic},
        )


def seed_corpus_record(num_documents: int, characters: int) -> SourceRecord:
    """Provenance for the synthetic corpus, labelled as synthetic."""
    return SourceRecord(
        name="bravien-seed-synthetic",
        url="local://bravien.data.download.generate_seed_corpus",
        license="CC0 1.0 (generated locally by this repository)",
        documents=num_documents,
        bytes=characters,
        processing=["template generation"],
        notes="Synthetic template text for offline smoke tests. A model trained "
        "only on this is a plumbing check, not a useful language model.",
    )


def fetch_all(
    names: list[str] | None = None,
    cache_dir: str | Path = "datasets/raw",
    *,
    allow_offline: bool = True,
) -> tuple[list[Path], list[SourceRecord], list[str]]:
    """Download several registry sources, tolerating individual failures.

    Returns the files fetched, their provenance records, and the errors seen. An
    empty file list with `allow_offline` set is not an error — the caller falls
    back to `generate_seed_corpus`.
    """
    selected = names or list(SOURCES)
    unknown = [n for n in selected if n not in SOURCES]
    if unknown:
        raise KeyError(f"unknown source(s): {unknown}. Known: {sorted(SOURCES)}")

    paths: list[Path] = []
    records: list[SourceRecord] = []
    errors: list[str] = []

    for name in selected:
        try:
            path, record = fetch_source(SOURCES[name], cache_dir)
            record.notes += f" checksum={file_checksum(path)}"
            paths.append(path)
            records.append(record)
        except DownloadError as exc:
            errors.append(str(exc))
            logger.warning("skipping %s: %s", name, exc)

    if not paths and not allow_offline:
        raise DownloadError(
            "no sources could be downloaded: " + "; ".join(errors)
        )
    return paths, records, errors
