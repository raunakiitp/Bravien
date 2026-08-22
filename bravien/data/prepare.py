"""End-to-end data preparation (§12).

Runs acquire -> clean -> filter -> deduplicate -> tokenize -> pack as one
streaming pass and writes a manifest describing exactly what happened. The
manifest is the deliverable as much as the `.bin` is: a token file with no
provenance cannot be defended (§14).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from bravien.data.clean import CleanOptions, Document, clean_text, looks_like_html
from bravien.data.dataset import contiguous_splits
from bravien.data.deduplicate import Deduplicator
from bravien.data.download import (
    fetch_all,
    generate_seed_corpus,
    load_directory,
    load_text_file,
    seed_corpus_record,
)
from bravien.data.filter import FilterStats, QualityThresholds, filter_document
from bravien.data.manifest import Manifest, SourceRecord, StageStats, file_checksum
from bravien.data.pack import count_packed_sequences
from bravien.data.tokenize import tokenize_to_file
from bravien.utils.logging import Timer, format_count, get_logger
from bravien.utils.paths import ensure_dir

if TYPE_CHECKING:
    from bravien.tokenizer.tokenizer import BravienTokenizer

logger = get_logger("data.prepare")


@dataclass
class PrepareConfig:
    """Everything the pipeline needs, in one auditable object."""

    output_dir: Path = Path("datasets/prepared")
    raw_dir: Path = Path("datasets/raw")
    name: str = "bravien-corpus"
    sequence_length: int = 1024

    #: Registry source names to download. None means "all of them".
    sources: list[str] | None = None
    #: A local directory of .txt/.jsonl files to include.
    local_dir: Path | None = None
    #: Fall back to the synthetic corpus when nothing else is available.
    allow_synthetic_fallback: bool = True
    synthetic_documents: int = 4000

    clean: CleanOptions = field(default_factory=CleanOptions)
    thresholds: QualityThresholds = field(default_factory=QualityThresholds)
    deduplicate: bool = True
    near_duplicates: bool = True
    dedup_threshold: float = 0.8
    redact_pii: bool = True
    seed: int = 0

    def __post_init__(self) -> None:
        # Coerce string paths: callers pass literals, and a str would fail later
        # at the first `/` join rather than here.
        self.output_dir = Path(self.output_dir)
        self.raw_dir = Path(self.raw_dir)
        if self.local_dir is not None:
            self.local_dir = Path(self.local_dir)
        if self.sequence_length <= 1:
            raise ValueError("sequence_length must be greater than 1")

    @property
    def token_path(self) -> Path:
        return self.output_dir / f"{self.name}.bin"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / f"{self.name}.manifest.json"


@dataclass
class PrepareResult:
    manifest: Manifest
    token_path: Path
    manifest_path: Path
    tokens: int
    sequences: int

    def summary(self) -> str:
        return self.manifest.summary()


def _acquire(config: PrepareConfig) -> tuple[Iterator[Document], list[SourceRecord]]:
    """Collect documents from every configured input.

    Returns a lazy iterator: nothing is read until the pipeline pulls on it.
    """
    records: list[SourceRecord] = []
    generators: list[Iterable[Document]] = []

    if config.local_dir is not None:
        local = Path(config.local_dir)
        if local.is_dir():
            generators.append(load_directory(local))
            records.append(
                SourceRecord(
                    name=f"local:{local.name}",
                    url=f"file://{local.resolve().as_posix()}",
                    license="unknown - user-supplied local files, review before "
                    "publishing a model trained on them",
                    processing=["read from disk"],
                )
            )
        else:
            logger.warning("local_dir %s does not exist; skipping", local)

    if config.sources is None or config.sources:
        paths, downloaded, errors = fetch_all(
            config.sources, config.raw_dir, allow_offline=True
        )
        for path in paths:
            generators.append(load_text_file(path))
        records.extend(downloaded)
        if errors:
            logger.warning(
                "%d source(s) unavailable (offline?): %s", len(errors), errors[0]
            )

    if not generators:
        if not config.allow_synthetic_fallback:
            raise RuntimeError(
                "no data available and synthetic fallback is disabled; either "
                "connect to the network or point local_dir at your own files"
            )
        logger.warning(
            "no real corpus available - falling back to %s synthetic documents. "
            "A model trained on these is a plumbing test only.",
            format_count(config.synthetic_documents),
        )
        generators.append(
            generate_seed_corpus(config.synthetic_documents, seed=config.seed)
        )
        records.append(seed_corpus_record(config.synthetic_documents, 0))

    def chained() -> Iterator[Document]:
        for gen in generators:
            yield from gen

    return chained(), records


def prepare_dataset(
    config: PrepareConfig, tokenizer: BravienTokenizer
) -> PrepareResult:
    """Run the whole pipeline and write the token file plus manifest."""
    ensure_dir(config.output_dir)
    documents, source_records = _acquire(config)

    clean_stats = StageStats(stage="clean")
    filter_stats = FilterStats()
    dedup = (
        Deduplicator(
            near_duplicates=config.near_duplicates,
            threshold=config.dedup_threshold,
            seed=config.seed,
        )
        if config.deduplicate
        else None
    )

    def staged() -> Iterator[Document]:
        """Clean, filter and dedupe in one pass over the input."""
        for doc in documents:
            clean_stats.documents_in += 1
            clean_stats.characters_in += len(doc.text)

            options = config.clean
            if not options.strip_html and looks_like_html(doc.text):
                # Only pay for tag stripping on documents that carry markup.
                options = CleanOptions(**{**vars(config.clean), "strip_html": True})
            doc.text = clean_text(doc.text, options)

            clean_stats.documents_out += 1
            clean_stats.characters_out += len(doc.text)

            kept = filter_document(
                doc, config.thresholds, filter_stats, redact=config.redact_pii
            )
            if kept is None:
                continue

            if dedup is not None:
                reason = dedup.is_duplicate(kept.text)
                if reason is not None:
                    continue

            yield kept

    logger.info("preparing %s -> %s", config.name, config.token_path)
    with Timer() as timer:
        info = tokenize_to_file(
            staged(), tokenizer, config.token_path, append_eos=True
        )
    logger.info(
        "tokenized %s tokens in %s", format_count(info["tokens"]), timer
    )

    filter_stage = StageStats(
        stage="filter",
        documents_in=filter_stats.seen,
        documents_out=filter_stats.kept,
        dropped=dict(filter_stats.dropped),
        notes=(
            f"pii redactions: {filter_stats.pii_redactions}"
            if filter_stats.pii_redactions
            else ""
        ),
    )

    stages = [clean_stats, filter_stage]
    if dedup is not None:
        stages.append(
            StageStats(
                stage="deduplicate",
                documents_in=dedup.stats.seen,
                documents_out=dedup.stats.kept,
                dropped=dedup.stats.dropped,
                notes=(
                    f"minhash near-dup threshold {config.dedup_threshold}"
                    if config.near_duplicates
                    else "exact only"
                ),
            )
        )

    tokens = int(info["tokens"])
    sequences = count_packed_sequences(tokens, config.sequence_length)

    manifest = Manifest(
        name=config.name,
        stage="prepared",
        documents=info["documents"],
        characters=info["characters"],
        tokens=tokens,
        sequences=sequences,
        sequence_length=config.sequence_length,
        languages=dict(filter_stats.languages),
        stages=stages,
        sources=source_records,
        files=[
            {
                "path": config.token_path.name,
                "bytes": config.token_path.stat().st_size,
                "checksum": file_checksum(config.token_path),
                "dtype": info["dtype"],
                "tokens": tokens,
            }
        ],
        tokenizer={
            "vocab_size": info["vocab_size"],
            "checksum": info["tokenizer_checksum"],
            "chars_per_token": info["chars_per_token"],
        },
        extra={
            "prepare_seconds": round(timer.elapsed, 2),
            "truncated_documents": info["truncated_documents"],
            "empty_documents": info["empty_documents"],
            "tokens_per_source": info["tokens_per_source"],
        },
    )
    manifest.record_environment()

    try:
        train, val = contiguous_splits(tokens)
        manifest.extra["splits"] = {
            "train": {"start": train.start, "end": train.end, "tokens": train.tokens},
            "validation": {"start": val.start, "end": val.end, "tokens": val.tokens},
        }
    except ValueError as exc:
        # Too small to hold out a validation split: record the fact instead of
        # inventing one.
        manifest.extra["splits_error"] = str(exc)

    manifest.content_checksum = manifest.checksum()
    manifest.save(config.manifest_path)

    return PrepareResult(
        manifest=manifest,
        token_path=config.token_path,
        manifest_path=config.manifest_path,
        tokens=tokens,
        sequences=sequences,
    )


def corpus_for_tokenizer(
    config: PrepareConfig, *, max_documents: int | None = None
) -> Iterator[str]:
    """Cleaned, filtered text for tokenizer training.

    The tokenizer must be trained before the corpus can be tokenized, so this
    runs the same early stages and yields raw strings.
    """
    documents, _ = _acquire(config)
    stats = FilterStats()
    emitted = 0
    for doc in documents:
        doc.text = clean_text(doc.text, config.clean)
        kept = filter_document(doc, config.thresholds, stats, redact=config.redact_pii)
        if kept is None:
            continue
        yield kept.text
        emitted += 1
        if max_documents is not None and emitted >= max_documents:
            return
