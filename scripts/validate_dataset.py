"""Dataset integrity and validation tool.

Verifies:
1. JSONL syntax and schema validity of all split files
2. Strict role ordering and structural invariants
3. No prompt or content overlap across train/val/test splits (Zero Leakage)
4. Absence of secrets, credentials, or dangerous patterns
5. Manifest checksum accuracy
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bravien.data.filter import contains_secrets
from bravien.data.schema import BravienTrainingExample, ValidationError
from bravien.utils.logging import configure_stdout, get_logger, setup_logging

logger = get_logger("scripts.validate_dataset")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a prepared Bravien dataset for integrity, correctness, and zero contamination.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-dir",
        default="data/processed",
        help="Directory containing train.jsonl, val.jsonl, and test.jsonl",
    )
    parser.add_argument(
        "--manifest",
        default="data/manifests/dataset_manifest.json",
        help="Path to dataset_manifest.json",
    )
    return parser


def validate_split_file(
    path: Path,
    split_name: str,
    seen_prompts: dict[str, str],
) -> tuple[int, list[str]]:
    """Validate one split file."""
    errors: list[str] = []
    count = 0

    if not path.exists():
        return 0, [f"Split file missing: {path}"]

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                errors.append(f"{split_name}:{line_num}: Empty line encountered")
                continue

            try:
                data = json.loads(line_str)
                ex = BravienTrainingExample.from_dict(data)
            except json.JSONDecodeError as exc:
                errors.append(f"{split_name}:{line_num}: JSON syntax error: {exc}")
                continue
            except ValidationError as exc:
                errors.append(f"{split_name}:{line_num}: Schema error: {exc}")
                continue

            count += 1

            # Check for secrets
            for m in ex.messages:
                has_sec, sec_type = contains_secrets(m.content)
                if has_sec:
                    errors.append(f"{split_name}:{line_num}: Secret detected ({sec_type}) in {m.role} turn")

            # Check prompt contamination across splits
            user_fp = ex.user_prompt_fingerprint()
            if user_fp in seen_prompts and seen_prompts[user_fp] != split_name:
                errors.append(
                    f"{split_name}:{line_num}: Data leakage! Prompt already exists in '{seen_prompts[user_fp]}' split."
                )
            else:
                seen_prompts[user_fp] = split_name

    return count, errors


def main(argv: list[str] | None = None) -> int:
    configure_stdout()
    setup_logging(level="INFO")
    args = build_parser().parse_args(argv)

    dataset_dir = Path(args.dataset_dir)
    manifest_path = Path(args.manifest)

    print("=======================================================")
    print(f"VALIDATING BRAVIEN DATASET AT: {dataset_dir}")
    print("=======================================================")

    all_errors: list[str] = []
    seen_prompts: dict[str, str] = {}
    counts: dict[str, int] = {}

    for split in ("train", "val", "test"):
        path = dataset_dir / f"{split}.jsonl"
        count, errors = validate_split_file(path, split, seen_prompts)
        counts[split] = count
        all_errors.extend(errors)
        print(f"[{split.upper():5s}] {count:,} samples checked ({len(errors)} issues)")

    # Validate Manifest if provided
    if manifest_path.exists():
        print(f"\nVerifying Manifest: {manifest_path}...")
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

            files_info = manifest.get("files", {})
            for split in ("train", "val", "test"):
                if split in files_info:
                    expected_sha = files_info[split].get("sha256")
                    actual_sha = file_sha256(dataset_dir / f"{split}.jsonl")
                    if expected_sha != actual_sha:
                        all_errors.append(
                            f"Manifest sha256 mismatch for {split}: expected {expected_sha}, got {actual_sha}"
                        )
                    expected_count = files_info[split].get("count")
                    if expected_count != counts.get(split):
                        all_errors.append(
                            f"Manifest count mismatch for {split}: expected {expected_count}, got {counts.get(split)}"
                        )
            print("Manifest checksum and sample count checks complete.")
        except Exception as exc:
            all_errors.append(f"Failed to read/verify manifest: {exc}")
    else:
        logger.warning(f"Manifest file not found at {manifest_path}")

    print("\n=======================================================")
    if all_errors:
        print(f"FAILED: {len(all_errors)} validation errors detected:")
        for err in all_errors[:20]:
            print(f"  - {err}")
        if len(all_errors) > 20:
            print(f"  ... and {len(all_errors) - 20} more")
        print("=======================================================\n")
        return 1
    else:
        print("SUCCESS: All dataset integrity, safety, and leakage checks passed!")
        print("=======================================================\n")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
