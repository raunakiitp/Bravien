"""Build the official pretraining dataset manifest for native Bravien pretraining."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from bravien.data.pretraining_manifest import CorpusSource, PretrainingManifest


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def estimate_tokens_in_file(filepath: Path) -> tuple[int, int]:
    """Return (estimated_token_count, sample_count)."""
    sample_count = 0
    total_chars = 0

    if filepath.suffix == ".jsonl":
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    sample_count += 1
                    total_chars += len(line)
    elif filepath.suffix == ".json":
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    sample_count = len(data)
                    total_chars = sum(len(str(x)) for x in data)
                elif isinstance(data, dict):
                    sample_count = len(data)
                    total_chars = len(json.dumps(data))
            except json.JSONDecodeError:
                sample_count = 1
                total_chars = filepath.stat().st_size
    else:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.strip():
                    sample_count += 1
                    total_chars += len(line)

    estimated_tokens = max(1, int(total_chars / 4.0))  # Standard 4 chars/token heuristic
    return estimated_tokens, sample_count


def infer_domain(filepath: Path) -> str:
    name = filepath.stem.lower()
    if "code" in name or "python" in name or "ts" in name:
        return "code"
    if "math" in name or "reason" in name:
        return "math_stem"
    if "hinglish" in name:
        return "hinglish"
    if "hindi" in name:
        return "multilingual_hindi"
    if "tool" in name or "agent" in name:
        return "agent_tools"
    if "safety" in name or "refusal" in name:
        return "safety"
    if "sft" in name or "instruction" in name or "stage2" in name:
        return "instruction_sft"
    return "general_web"


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan and build Bravien pretraining data manifest.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing dataset files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="manifests/pretraining_manifest.json",
        help="Output JSON path for the manifest.",
    )

    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("BUILDING BRAVIEN PRETRAINING DATA MANIFEST")
    print("=" * 65)

    data_path = Path(args.data_dir)
    manifest = PretrainingManifest(
        manifest_version="1.0.0",
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        metadata={"generator": "scripts/build_pretraining_manifest.py"},
    )

    if data_path.exists():
        found_files = sorted(list(data_path.glob("*.jsonl")) + list(data_path.glob("*.json")) + list(data_path.glob("*.txt")))
        print(f"Discovered {len(found_files)} data shard(s) in '{data_path}'...")

        for file_path in found_files:
            tokens, samples = estimate_tokens_in_file(file_path)
            sha = compute_sha256(file_path)
            domain = infer_domain(file_path)

            source = CorpusSource(
                source_id=f"src_{file_path.stem}",
                name=file_path.name,
                domain=domain,  # type: ignore
                language="hinglish" if domain == "hinglish" else ("code" if domain == "code" else "en"),
                estimated_tokens=tokens,
                sample_count=samples,
                license="Apache-2.0 / Permissive",
                file_path=str(file_path.relative_to(Path.cwd())),
                sha256_checksum=sha,
                quality_score=0.98,
            )
            manifest.add_source(source)
            print(f"  + [{domain:<15}] {file_path.name:<30} -> {samples:>6,} samples | ~{tokens:>9,} tokens")
    else:
        print(f"Notice: Data directory '{data_path}' not found or empty.")

    out_file = Path(args.output)
    manifest.save(out_file)

    print("-" * 65)
    print(f"Total Sources:          {len(manifest.sources)}")
    print(f"Total Samples:          {manifest.total_samples:,}")
    print(f"Total Estimated Tokens: ~{manifest.total_estimated_tokens:,}")
    print(f"Manifest written to:    {out_file}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
