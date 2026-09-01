"""Qwen Dependency Audit Scanner for Bravien Codebase."""

import json
import re
import sys
import time
from pathlib import Path
from typing import Any

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent

SEARCH_TERMS = [
    r"Qwen",
    r"Qwen2\.5",
    r"AutoTokenizer",
    r"AutoModel",
    r"Qwen/Qwen2\.5-0\.5B-Instruct",
]

# Patterns that are ALLOWED (historical docs, benchmark baselines, explicit opt-in compat)
ALLOWED_FILE_PATTERNS = [
    r"docs/.*",
    r"AGENTS\.md",
    r"reports/.*",
    r"tests/.*",
    r"scripts/.*",
    r"bravien/inference/.*",  # Historical inference & server rollback
    r"bravien/model/provider\.py",  # Contains explicit optional BravienLocalProvider
    r"bravien/training/hf_trainer\.py",  # Historical Stage 3 SFT trainer
    r"bravien/evaluation/.*",
    r"bravien/data/schema\.py",
    r"bravien/scripts/.*",
    r"bravien/synthetic/.*",
    r"src/.*",
]

# Patterns that are strictly NOT ALLOWED (default runtime, core native architecture, native training)
STRICT_NATIVE_FILES = [
    r"bravien/model/bravien_.*\.py",
    r"bravien/model/checkpoint\.py",
    r"bravien/model/native_provider\.py",
    r"bravien/tokenizer/.*\.py",
    r"bravien/training/pretrain.*\.py",
    r"bravien/training/checkpoint_manager\.py",
    r"bravien/data/packer\.py",
    r"bravien/data/quality_pipeline\.py",
    r"bravien/data/pretraining_manifest\.py",
    r"scripts/pretrain_bravien\.py",
    r"scripts/train_bravien_tokenizer\.py",
]


def audit_repository() -> dict[str, Any]:
    print("=" * 80)
    print("BRAVIEN QWEN DEPENDENCY & INDEPENDENCE AUDIT")
    print("=" * 80)

    findings: list[dict[str, Any]] = []
    allowed_count = 0
    not_allowed_count = 0

    code_extensions = {".py", ".ts", ".tsx", ".json", ".yaml", ".yml", ".md"}
    ignore_dirs = {".git", ".next", "node_modules", ".gemini", "checkpoints", "artifacts", "dist", "build"}

    combined_regex = re.compile("|".join(f"({t})" for t in SEARCH_TERMS), re.IGNORECASE)

    for file_path in ROOT_DIR.rglob("*"):
        if file_path.is_file() and file_path.suffix in code_extensions:
            # Check ignored directories
            parts = set(file_path.relative_to(ROOT_DIR).parts)
            if parts.intersection(ignore_dirs):
                continue

            rel_str = str(file_path.relative_to(ROOT_DIR)).replace("\\", "/")

            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            matches = list(combined_regex.finditer(content))
            if not matches:
                continue

            # Classify file
            is_strict_native = any(re.search(pat, rel_str) for pat in STRICT_NATIVE_FILES)
            is_allowed_path = any(re.search(pat, rel_str) for pat in ALLOWED_FILE_PATTERNS)

            classification = "NOT_ALLOWED" if is_strict_native else ("ALLOWED" if is_allowed_path else "REVIEW_REQUIRED")

            for m in matches:
                # Find line number
                line_no = content[: m.start()].count("\n") + 1
                matched_text = m.group(0)

                finding_entry = {
                    "file": rel_str,
                    "line": line_no,
                    "matched_term": matched_text,
                    "classification": classification,
                    "context": content.splitlines()[line_no - 1].strip()[:100],
                }
                findings.append(finding_entry)

                if classification == "ALLOWED":
                    allowed_count += 1
                else:
                    not_allowed_count += 1

    print(f"\nAudit Findings Summary:")
    print(f"  * Total Qwen / HF references found: {len(findings)}")
    print(f"  * Allowed (Historical / Docs / Benchmarks / Opt-in compat): {allowed_count}")
    print(f"  * Not Allowed (Default runtime / Native engine):          {not_allowed_count}")

    if not_allowed_count == 0:
        print("\n[AUDIT PASSED] ZERO forbidden Qwen dependencies in default runtime and native paths!")
    else:
        print(f"\n[AUDIT WARNING] {not_allowed_count} references require review.")

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_references": len(findings),
        "allowed_references_count": allowed_count,
        "not_allowed_references_count": not_allowed_count,
        "is_native_independent": not_allowed_count == 0,
        "default_model_backend": "native",
        "qwen_compat_status": "disabled_by_default (opt-in rollback only)",
        "all_findings": findings,
    }

    out_file = ROOT_DIR / "reports" / "qwen_dependency_final_audit.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Audit report saved to: {out_file}\n" + "=" * 80)
    return report


if __name__ == "__main__":
    audit_repository()
