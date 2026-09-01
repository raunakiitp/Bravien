"""Pretraining Dataset Packing CLI.

Processes modular training corpus (English, Hindi, Hinglish, Code, Math, Technical QA)
using the native BravienTokenizer and packs sequences into fixed-length contiguous token blocks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import torch

from bravien.data.packer import PackedPretrainingDataset
from bravien.data.quality_pipeline import DocumentQualityFilter
from bravien.tokenizer.tokenizer import BravienTokenizer


def stream_corpus_texts(data_dir: Path) -> Iterator[str]:
    """Stream clean texts from all JSONL and JSON files in data directory."""
    filter_engine = DocumentQualityFilter()

    # Search for all jsonl and json files
    files = list(data_dir.rglob("*.jsonl")) + list(data_dir.rglob("*.json"))
    for file_path in files:
        if "manifest" in file_path.name or "package" in file_path.name or "report" in file_path.name:
            continue
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                if file_path.suffix == ".jsonl":
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            # Handle conversation format or raw text
                            if "messages" in obj:
                                full_text = "\n".join(
                                    f"{m.get('role', 'user')}: {m.get('content', '')}"
                                    for m in obj["messages"]
                                )
                                passed, _ = filter_engine.filter_document(full_text, doc_id=file_path.name)
                                if passed:
                                    yield full_text
                            elif "text" in obj:
                                passed, _ = filter_engine.filter_document(obj["text"], doc_id=file_path.name)
                                if passed:
                                    yield obj["text"]
                            elif "instruction" in obj and "response" in obj:
                                full_text = f"User: {obj['instruction']}\nAssistant: {obj['response']}"
                                passed, _ = filter_engine.filter_document(full_text, doc_id=file_path.name)
                                if passed:
                                    yield full_text
                        except Exception:
                            continue
                elif file_path.suffix == ".json":
                    try:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict) and "text" in item:
                                    yield item["text"]
                    except Exception:
                        continue
        except Exception:
            continue


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack raw text datasets into contiguous token blocks.")
    parser.add_argument("--data-dir", type=str, default="data", help="Input directory containing raw dataset files.")
    parser.add_argument("--tokenizer-dir", type=str, default="tokenizers/bravien-native", help="Path to native tokenizer.")
    parser.add_argument("--output-file", type=str, default="data/packed_pretrain/train_packed.pt", help="Path to output .pt file.")
    parser.add_argument("--seq-len", type=int, default=1024, help="Target sequence block length.")
    parser.add_argument("--max-blocks", type=int, default=500, help="Maximum blocks to pack.")

    args = parser.parse_args()

    print("=" * 65)
    print("BRAVIEN PRETRAINING DATA PACKER")
    print("=" * 65)
    print(f"Data Source Directory: {args.data_dir}")
    print(f"Tokenizer Directory:   {args.tokenizer_dir}")
    print(f"Target Block Length:   {args.seq_len} tokens")
    print(f"Output File:           {args.output_file}")

    tok = BravienTokenizer.from_pretrained(Path(args.tokenizer_dir))
    eos_id = tok.eos_token_id

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    blocks: list[torch.Tensor] = []
    current_tokens: list[int] = []
    doc_count = 0
    total_tokens = 0

    print("\nTokenizing and packing documents into contiguous blocks...")
    for text in stream_corpus_texts(Path(args.data_dir)):
        tokens = tok.encode(text, add_special_tokens=False)
        if not tokens:
            continue
        tokens.append(eos_id)
        current_tokens.extend(tokens)
        doc_count += 1
        total_tokens += len(tokens)

        while len(current_tokens) >= args.seq_len:
            block = torch.tensor(current_tokens[: args.seq_len], dtype=torch.long)
            blocks.append(block)
            current_tokens = current_tokens[args.seq_len :]
            if len(blocks) >= args.max_blocks:
                break

        if len(blocks) >= args.max_blocks:
            break

    # If remaining tokens exist, pad to seq_len
    if current_tokens and len(blocks) < args.max_blocks:
        padded = current_tokens + [tok.pad_token_id] * (args.seq_len - len(current_tokens))
        blocks.append(torch.tensor(padded[: args.seq_len], dtype=torch.long))

    dataset = PackedPretrainingDataset(blocks, max_seq_len=args.seq_len)
    torch.save(dataset, out_path)

    print(f"\n[PASS] Packing Complete:")
    print(f"  - Total Documents Processed: {doc_count:,}")
    print(f"  - Total Raw Tokens:          {total_tokens:,}")
    print(f"  - Packed Blocks Generated:   {len(blocks):,}")
    print(f"  - Saved to:                  {out_path}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
