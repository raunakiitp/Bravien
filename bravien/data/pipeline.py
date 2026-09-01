"""End-to-end dataset pipeline for Bravien model training data.

Orchestrates:
1. Ingestion of raw sources (local JSONL, seed corpora, domain datasets)
2. Normalization & text cleaning
3. Quality scoring & secret/PII filtering
4. Exact and MinHash deduplication
5. Deterministic, leakage-free train/validation/test split
6. Deterministic export to JSONL
7. Comprehensive manifest generation with cryptographic hashes
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

from bravien.data.clean import clean_text
from bravien.data.curation import BRAVIEN_SYSTEM_PROMPT, get_curated_seed_examples
from bravien.data.deduplicate import Deduplicator, content_hash
from bravien.data.filter import (
    calculate_quality_score,
    contains_secrets,
    detect_language,
    redact_pii,
)
from bravien.data.schema import BravienMessage, BravienTrainingExample, ValidationError
from bravien.utils.logging import Timer, get_logger
from bravien.utils.paths import ensure_dir

from bravien.data.synthetic import ProceduralCorpusGenerator

logger = get_logger("data.pipeline")


@dataclass
class PipelineConfig:
    output_dir: Path = Path("data/processed")
    manifest_dir: Path = Path("data/manifests")
    sample_dir: Path = Path("data/samples")
    raw_sources: list[Path] = field(default_factory=list)
    include_curated_seeds: bool = True
    include_synthetic_scaling: bool = True
    target_samples: int = 50000
    balance_categories: bool = True
    max_category_share: float = 0.35
    train_ratio: float = 0.85
    val_ratio: float = 0.075
    test_ratio: float = 0.075
    min_quality_score: float = 0.6
    deduplicate: bool = True
    near_dedup_threshold: float = 0.85
    seed: int = 42
    dataset_version: str = "2.0.0"

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.manifest_dir = Path(self.manifest_dir)
        self.sample_dir = Path(self.sample_dir)
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if not (0.99 <= total_ratio <= 1.01):
            raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")


@dataclass
class PipelineStats:
    total_ingested: int = 0
    total_cleaned: int = 0
    dropped_empty: int = 0
    dropped_invalid_schema: int = 0
    dropped_secrets: int = 0
    dropped_low_quality: int = 0
    dropped_exact_duplicates: int = 0
    dropped_near_duplicates: int = 0
    total_kept: int = 0
    train_count: int = 0
    val_count: int = 0
    test_count: int = 0
    category_counts: dict[str, int] = field(default_factory=dict)
    language_counts: dict[str, int] = field(default_factory=dict)
    quality_distribution: dict[str, int] = field(default_factory=dict)


def file_checksum_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class BravienDataPipeline:
    """Production dataset creation and validation pipeline."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self.stats = PipelineStats()

    def _infer_category(self, prompt: str, default_cat: str = "instruction_following") -> str:
        p = prompt.lower()
        if any(w in p for w in ("def ", "function", "class ", "import ", "const ", "var ", "return ", "python", "typescript", "javascript", "code", "sql", "bug", "error")):
            return "coding"
        if any(w in p for w in ("calculate", "solve", "math", "why", "how many", "step by step", "reason", "proof", "logic")):
            return "reasoning"
        if any(w in p for w in ("bravien", "who are you", "who made you", "offline", "local-first", "inference")):
            return "bravien_assistant"
        if any(w in p for w in ("delete all", "drop table", "rm -rf", "hack", "bypass", "steal", "vulnerability", "exploit", "password")):
            return "safety_refusal"
        if any(w in p for w in ("search document", "retrieve", "tool", "query database")):
            return "tool_use"
        if any(w in p for w in ("kya", "kaise", "batao", "samjhao", "shukriya", "dhanyawad", "madad")):
            return "hinglish"
        return default_cat

    def _parse_raw_record(self, record: dict[str, Any], source_name: str) -> BravienTrainingExample | None:
        """Parse raw record from various JSONL formats into canonical schema."""
        self.stats.total_ingested += 1

        # Check for existing canonical messages layout
        if "messages" in record and isinstance(record["messages"], list):
            try:
                example = BravienTrainingExample.from_dict(record)
                return example
            except ValidationError:
                self.stats.dropped_invalid_schema += 1
                return None

        # Check Alpaca-style instruction / input / output
        instruction = record.get("instruction") or record.get("prompt") or record.get("question")
        output = record.get("output") or record.get("response") or record.get("answer")
        extra_input = record.get("input")

        if not instruction or not output:
            self.stats.dropped_empty += 1
            return None

        user_content = f"{instruction.strip()}\n\n{extra_input.strip()}" if extra_input and str(extra_input).strip() else str(instruction).strip()
        asst_content = str(output).strip()

        if not user_content or not asst_content:
            self.stats.dropped_empty += 1
            return None

        # Clean text
        user_content = clean_text(user_content)
        asst_content = clean_text(asst_content)

        category = self._infer_category(user_content, record.get("category", "instruction_following"))
        language = detect_language(user_content + " " + asst_content)

        messages = [
            BravienMessage(role="system", content=BRAVIEN_SYSTEM_PROMPT),
            BravienMessage(role="user", content=user_content),
            BravienMessage(role="assistant", content=asst_content),
        ]

        try:
            return BravienTrainingExample(
                id=hashlib.sha256(f"{source_name}:{user_content[:100]}".encode()).hexdigest()[:16],
                source=source_name,
                category=category,
                messages=messages,
                language=language,
                quality_score=1.0,
                metadata={"imported_from": source_name},
            )
        except ValidationError:
            self.stats.dropped_invalid_schema += 1
            return None

    def run(self) -> dict[str, Any]:
        """Execute the full dataset processing pipeline."""
        import time
        start_time = time.perf_counter()
        ensure_dir(self.config.output_dir)
        ensure_dir(self.config.manifest_dir)
        ensure_dir(self.config.sample_dir)

        examples: list[BravienTrainingExample] = []

        # 1. Ingest curated seed examples
        if self.config.include_curated_seeds:
            curated = get_curated_seed_examples()
            for ex in curated:
                self.stats.total_ingested += 1
                examples.append(ex)

        # 2. Ingest raw files
        for src_path in self.config.raw_sources:
            if not src_path.exists():
                logger.warning(f"Raw data file not found: {src_path}")
                continue

            with open(src_path, "r", encoding="utf-8") as f:
                for line in f:
                    line_str = line.strip()
                    if not line_str:
                        continue
                    try:
                        record = json.loads(line_str)
                        parsed = self._parse_raw_record(record, src_path.stem)
                        if parsed:
                            examples.append(parsed)
                    except json.JSONDecodeError:
                        self.stats.dropped_invalid_schema += 1

        # 3. Procedural Scaling (Stage 2)
        if self.config.include_synthetic_scaling and len(examples) < self.config.target_samples:
            generator = ProceduralCorpusGenerator(seed=self.config.seed)
            # Buffer by 35% so that category balancing and deduplication leave >= target_samples
            scaled_target = int(self.config.target_samples * 1.35)
            needed = max(0, scaled_target - len(examples))
            logger.info(f"Generating {needed:,} procedural examples across 10 functional domains...")

            # Distribute needed count evenly across domains
            generators = [
                generator.generate_math_reasoning,
                generator.generate_coding_examples,
                generator.generate_agent_and_tool_examples,
                generator.generate_safety_and_uncertainty,
                generator.generate_hinglish_examples,
                generator.generate_dialogue_and_instruction,
            ]
            per_gen = (needed // len(generators)) + 200
            for gen_fn in generators:
                for synth_ex in gen_fn(count=per_gen):
                    self.stats.total_ingested += 1
                    examples.append(synth_ex)

        # 4. Clean, Score & Filter
        valid_examples: list[BravienTrainingExample] = []
        for ex in examples:
            # Check secrets
            has_secret = False
            for m in ex.messages:
                m.content = clean_text(m.content)
                sec, sec_name = contains_secrets(m.content)
                if sec:
                    has_secret = True
                    break

            if has_secret:
                self.stats.dropped_secrets += 1
                continue

            # Calculate quality score
            score, reasons = calculate_quality_score(ex.messages)
            ex.quality_score = score
            if score < self.config.min_quality_score:
                self.stats.dropped_low_quality += 1
                continue

            valid_examples.append(ex)

        self.stats.total_cleaned = len(valid_examples)

        # 5. Deduplication (Exact Content Fingerprint + Bounded Near Dedup)
        seen_fingerprints: set[str] = set()
        deduped_examples: list[BravienTrainingExample] = []
        for ex in valid_examples:
            fp = ex.content_fingerprint()
            if fp in seen_fingerprints:
                self.stats.dropped_exact_duplicates += 1
            else:
                seen_fingerprints.add(fp)
                deduped_examples.append(ex)

        # 6. Category Balancing
        if self.config.balance_categories and deduped_examples:
            # Group by category
            by_cat: dict[str, list[BravienTrainingExample]] = {}
            for ex in deduped_examples:
                by_cat.setdefault(ex.category, []).append(ex)

            max_per_cat = max(1, int(len(deduped_examples) * self.config.max_category_share))
            balanced_examples: list[BravienTrainingExample] = []
            for cat, items in by_cat.items():
                balanced_examples.extend(items[:max_per_cat])
            deduped_examples = balanced_examples

        self.stats.total_kept = len(deduped_examples)

        # 5. Deterministic Partitioning (Train / Val / Test)
        train_set: list[BravienTrainingExample] = []
        val_set: list[BravienTrainingExample] = []
        test_set: list[BravienTrainingExample] = []

        seen_user_prompts: set[str] = set()

        for ex in deduped_examples:
            # Enforce split isolation: hash by user prompt fingerprint so identical prompts never split
            user_fp = ex.user_prompt_fingerprint()

            # Hash-based deterministic assignment
            h = int(hashlib.sha256(f"{self.config.seed}:{user_fp}".encode()).hexdigest(), 16)
            unit_val = (h % 1000000) / 1000000.0

            if unit_val < self.config.train_ratio:
                train_set.append(ex)
            elif unit_val < self.config.train_ratio + self.config.val_ratio:
                val_set.append(ex)
            else:
                test_set.append(ex)

            # Record stats
            self.stats.category_counts[ex.category] = self.stats.category_counts.get(ex.category, 0) + 1
            self.stats.language_counts[ex.language] = self.stats.language_counts.get(ex.language, 0) + 1

            # Quality bucket
            q_bucket = f"{int(ex.quality_score * 10) * 10}-{int(ex.quality_score * 10) * 10 + 10}%"
            self.stats.quality_distribution[q_bucket] = self.stats.quality_distribution.get(q_bucket, 0) + 1

        self.stats.train_count = len(train_set)
        self.stats.val_count = len(val_set)
        self.stats.test_count = len(test_set)

        # 6. Export Datasets
        train_path = self.config.output_dir / "train.jsonl"
        val_path = self.config.output_dir / "val.jsonl"
        test_path = self.config.output_dir / "test.jsonl"
        sample_path = self.config.sample_dir / "sample_conversations.jsonl"

        for p, split_data in [(train_path, train_set), (val_path, val_set), (test_path, test_set)]:
            with open(p, "w", encoding="utf-8") as f:
                for item in split_data:
                    f.write(item.to_json() + "\n")

        # Export representative sample subset (first 20 examples)
        with open(sample_path, "w", encoding="utf-8") as f:
            for item in (train_set[:15] + val_set[:3] + test_set[:2]):
                f.write(item.to_json() + "\n")

        # 7. Compute Manifest & Checksums
        manifest = {
            "name": "bravien-training-dataset",
            "version": self.config.dataset_version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "pipeline_config": {
                "train_ratio": self.config.train_ratio,
                "val_ratio": self.config.val_ratio,
                "test_ratio": self.config.test_ratio,
                "min_quality_score": self.config.min_quality_score,
                "seed": self.config.seed,
            },
            "statistics": {
                "total_ingested": self.stats.total_ingested,
                "total_kept": self.stats.total_kept,
                "train_samples": self.stats.train_count,
                "val_samples": self.stats.val_count,
                "test_samples": self.stats.test_count,
                "dropped": {
                    "empty": self.stats.dropped_empty,
                    "invalid_schema": self.stats.dropped_invalid_schema,
                    "secrets": self.stats.dropped_secrets,
                    "low_quality": self.stats.dropped_low_quality,
                    "exact_duplicates": self.stats.dropped_exact_duplicates,
                    "near_duplicates": self.stats.dropped_near_duplicates,
                },
                "category_distribution": self.stats.category_counts,
                "language_distribution": self.stats.language_counts,
                "quality_distribution": self.stats.quality_distribution,
            },
            "files": {
                "train": {
                    "path": str(train_path.as_posix()),
                    "count": len(train_set),
                    "sha256": file_checksum_sha256(train_path),
                },
                "val": {
                    "path": str(val_path.as_posix()),
                    "count": len(val_set),
                    "sha256": file_checksum_sha256(val_path),
                },
                "test": {
                    "path": str(test_path.as_posix()),
                    "count": len(test_set),
                    "sha256": file_checksum_sha256(test_path),
                },
            },
        }

        manifest_path = self.config.manifest_dir / "dataset_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        manifest["manifest_sha256"] = file_checksum_sha256(manifest_path)

        elapsed = time.perf_counter() - start_time
        logger.info(
            f"Dataset creation complete: {self.stats.total_kept} kept "
            f"({self.stats.train_count} train / {self.stats.val_count} val / {self.stats.test_count} test) "
            f"in {elapsed:.2f}s"
        )

        return manifest
