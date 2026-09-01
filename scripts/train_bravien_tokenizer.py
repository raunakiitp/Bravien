"""Train the native Bravien Byte-level BPE Tokenizer on curated corpus.

Supports English, Hindi, Hinglish, Code, JSON, Markdown, and Math.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterator

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from bravien.tokenizer.train import train_tokenizer


def stream_corpus_from_files(file_paths: list[Path]) -> Iterator[str]:
    """Stream text lines/documents from list of text/JSONL/JSON files."""
    for p in file_paths:
        if not p.exists():
            print(f"Warning: Corpus file '{p}' not found. Skipping.", file=sys.stderr)
            continue
        if p.suffix == ".jsonl":
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if "text" in data:
                            yield data["text"]
                        elif "prompt" in data and "response" in data:
                            yield f"{data['prompt']}\n{data['response']}"
                        elif "content" in data:
                            yield data["content"]
                        elif "messages" in data:
                            for msg in data["messages"]:
                                yield msg.get("content", "")
                    except json.JSONDecodeError:
                        yield line
        elif p.suffix == ".json":
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            yield item.get("text", item.get("content", str(item)))
                        else:
                            yield str(item)
                elif isinstance(data, dict):
                    for v in data.values():
                        yield str(v)
        else:
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.strip():
                        yield line


def get_default_training_corpus() -> list[str]:
    """Curated seed text covering English, Hindi, Hinglish, Code, Math, Tools, and Markdown."""
    return [
        # English Conversational & Systems
        "Bravien is a local-first sovereign artificial intelligence assistant designed for privacy and speed.",
        "The operating system utilizes multi-threading, asynchronous I/O, and non-blocking event loops.",
        "Distributed computing clusters communicate using remote procedure calls and protocol buffers.",
        "Memory management in systems programming requires careful tracking of heap allocations and stack frames.",

        # Hinglish & Hindi Multilingual
        "Python me list aur tuple me kya difference hota hai? List mutable hoti hai aur tuple immutable hota hai.",
        "Aapka task complete ho gaya hai. Ab aap next step ke liye prepare kar sakte hain.",
        "Kya aap mujhe batayenge ki database indexing kaise kaam karti hai? Ye query execution speed ko optimize karti hai.",
        "नमस्ते! ब्राविएन एक स्वतंत्र और स्थानीय कृत्रिम बुद्धिमत्ता सहायक है।",
        "गणित और कंप्यूटर विज्ञान में एल्गोरिदम का बहुत बड़ा महत्व है।",

        # Python Code
        "def quicksort(arr: list[int]) -> list[int]:\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)",
        "async def fetch_user_data(user_id: str, session: aiohttp.ClientSession) -> dict[str, Any]:\n    async with session.get(f'/api/users/{user_id}') as response:\n        return await response.json()",

        # TypeScript / JavaScript Code
        "interface AgentState {\n  sessionId: string;\n  turnIndex: number;\n  memoryStore: Map<string, string>;\n  tools: ToolDefinition[];\n}",
        "export const calculateMetric = (values: number[]): number => values.reduce((a, b) => a + b, 0) / values.length;",

        # JSON, XML & Tool Calling
        '{"tool": "calculator", "arguments": {"expression": "345 * 18"}, "status": "pending"}',
        '{"name": "unit_converter", "from_unit": "celsius", "to_unit": "fahrenheit", "value": 100}',
        '<TOOL_CALL>name="search_database", query="SELECT * FROM users WHERE active=1"</TOOL_CALL>',

        # Mathematics & LaTeX
        "Let f(x) = x^2 + 2x + 1. The derivative with respect to x is f'(x) = 2x + 2.",
        "The standard normal distribution is given by the probability density function \\frac{1}{\\sqrt{2\\pi}} e^{-\\frac{x^2}{2}}.",
        "Calculate the eigenvalues and eigenvectors of the 2x2 matrix A = [[4, 1], [2, 3]].",

        # Markdown & Structure
        "# Project Roadmap\n\n## Objectives\n- High throughput inference\n- Zero cloud dependencies\n- Strict data isolation",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a native Bravien tokenizer.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tokenizers/bravien-native",
        help="Directory where tokenizer.json and metadata will be saved.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32000,
        help="Target vocabulary size (including special tokens).",
    )
    parser.add_argument(
        "--corpus-files",
        nargs="*",
        default=[],
        help="Optional paths to raw text or JSONL corpus files.",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum token frequency for BPE merges.",
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("TRAINING BRAVIEN NATIVE TOKENIZER")
    print("=" * 60)
    print(f"Target Vocabulary Size: {args.vocab_size:,}")
    print(f"Output Directory:       {args.output_dir}")

    # Gather texts
    if args.corpus_files:
        files = [Path(p) for p in args.corpus_files]
        print(f"Reading from {len(files)} corpus file(s)...")
        text_stream = stream_corpus_from_files(files)
    else:
        # Check if Stage 2 data is available
        data_dir = Path("data")
        stage2_files = list(data_dir.glob("*.jsonl")) + list(data_dir.glob("*.json"))
        if stage2_files:
            print(f"Auto-discovered {len(stage2_files)} data files in data/ directory.")
            # Blend stage 2 files with seed corpus
            def blended_stream():
                for seed in get_default_training_corpus():
                    # Duplicate seed texts to ensure proper representation
                    for _ in range(5):
                        yield seed
                yield from stream_corpus_from_files(stage2_files)
            text_stream = blended_stream()
        else:
            print("No custom corpus files provided. Using curated multilingual seed corpus.")
            def seed_stream():
                for seed in get_default_training_corpus():
                    for _ in range(20):
                        yield seed
            text_stream = seed_stream()

    t0 = time.perf_counter()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = train_tokenizer(
        texts=text_stream,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        output_dir=out_dir,
        sources=[{"type": "bravien_curated_corpus", "version": "1.0.0"}],
    )
    elapsed = time.perf_counter() - t0

    print(f"\nTokenizer training complete in {elapsed:.2f}s!")
    print(f"Saved artifacts:")
    print(f"  - {out_dir / 'tokenizer.json'}")
    print(f"  - {out_dir / 'tokenizer_metadata.json'}")
    print(f"Actual vocabulary size: {tokenizer.vocab_size:,}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
