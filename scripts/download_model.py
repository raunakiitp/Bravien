"""Download a pretrained Hugging Face model for offline local use with Bravien.

Usage:
    python scripts/download_model.py --model Qwen/Qwen2.5-0.5B-Instruct
    python scripts/download_model.py --model HuggingFaceTB/SmolLM-135M-Instruct
"""

import argparse
import sys
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and cache a Hugging Face model locally.")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Hugging Face model ID to download.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional destination directory to save the model weights and tokenizer.",
    )
    args = parser.parse_args()

    print(f"Downloading model: {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True)

    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(out)
        model.save_pretrained(out)
        print(f"Successfully saved {args.model} to {out}")
    else:
        print(f"Successfully downloaded and cached {args.model}!")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
