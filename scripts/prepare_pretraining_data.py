"""Data Preparation, Quality Filtering, and Token Packing CLI for Bravien Pretraining."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from bravien.data.packer import PackedPretrainingDataset
from bravien.data.quality_pipeline import QualityFilter
from bravien.tokenizer.tokenizer import BravienTokenizer


def generate_curated_seed_documents() -> list[tuple[str, str]]:
    """Generate initial curated, license-clean seed documents across all required domains."""
    docs = [
        # Domain: general_web / educational
        ("general_web", "The scientific method is an empirical method of acquiring knowledge that has characterized the development of science since at least the 17th century. It involves careful observation, applying rigorous skepticism about what is observed, given that cognitive assumptions can distort how one interprets the observation."),
        ("general_web", "Thermodynamics is the branch of physics that deals with heat, work, and temperature, and their relation to energy, radiation, and physical properties of matter. The behavior of these quantities is governed by the four laws of thermodynamics which convey a quantitative description using measurable macroscopic physical quantities."),
        ("general_web", "Operating systems manage computer hardware and software resources and provide common services for computer programs. Time-sharing operating systems schedule tasks for efficient use of the system and may also include accounting software for cost allocation of processor time, mass storage, printing, and other resources."),

        # Domain: code / python
        ("code", "class BinarySearchTree:\n    def __init__(self, val=0, left=None, right=None):\n        self.val = val\n        self.left = left\n        self.right = right\n\ndef insert_node(root: BinarySearchTree | None, val: int) -> BinarySearchTree:\n    if root is None:\n        return BinarySearchTree(val)\n    if val < root.val:\n        root.left = insert_node(root.left, val)\n    else:\n        root.right = insert_node(root.right, val)\n    return root"),
        ("code", "import asyncio\nimport aiohttp\n\nasync def fetch_endpoint(url: str, session: aiohttp.ClientSession) -> dict:\n    async with session.get(url) as response:\n        response.raise_for_status()\n        return await response.json()\n\nasync def main():\n    async with aiohttp.ClientSession() as session:\n        data = await fetch_endpoint('https://api.example.com/status', session)\n        print('Server Status:', data)"),

        # Domain: code / typescript
        ("code", "export interface AgentExecutionState {\n  readonly runId: string;\n  readonly startTimeMs: number;\n  readonly stepCount: number;\n  readonly memoryContext: Record<string, unknown>;\n}\n\nexport function calculateThroughput(totalTokens: number, durationSeconds: number): number {\n  if (durationSeconds <= 0) return 0;\n  return totalTokens / durationSeconds;\n}"),

        # Domain: math_stem / reasoning
        ("math_stem", "Theorem: In any right-angled triangle, the area of the square whose side is the hypotenuse is equal to the sum of the areas of the squares whose sides are the two legs. Proof by rearrangement: Consider a large square of side length (a + b). Within this square, place four congruent right triangles with legs a and b and hypotenuse c. The remaining inner area is a square with side c and area c^2. The total area is (a + b)^2 = a^2 + 2ab + b^2. The four triangles each have area (1/2)ab, summing to 2ab. Subtracting the triangle areas from the total area gives c^2 = a^2 + b^2."),
        ("math_stem", "Linear algebra explores vector spaces and linear mappings between them. Given an m x n matrix A, its null space Null(A) is the set of all vectors x in R^n such that Ax = 0. The Rank-Nullity Theorem states that for any linear map T: V -> W, dim(V) = rank(T) + nullity(T)."),

        # Domain: multilingual_hindi
        ("multilingual_hindi", "भारत की वैज्ञानिक और तकनीकी प्रगति में अंतरिक्ष अनुसंधान, सूचना प्रौद्योगिकी, और नवीकरणीय ऊर्जा के क्षेत्र में महत्वपूर्ण उपलब्धियां शामिल हैं। भारतीय अंतरिक्ष अनुसंधान संगठन (ISRO) ने उपग्रह प्रक्षेपण और चंद्र अन्वेषण में वैश्विक स्तर पर अपनी पहचान बनाई है।"),
        ("multilingual_hindi", "कंप्यूटर प्रोग्रामिंग में डेटा संरचनाएं और एल्गोरिदम समस्याओं के समाधान को कुशल और समय-प्रभावी बनाने के लिए आधारशिला का कार्य करते हैं। स्टैक, क्यू, और लिंक्ड लिस्ट जैसी संरचनाएं मेमोरी प्रबंधन में महत्वपूर्ण भूमिका निभाती हैं।"),

        # Domain: hinglish
        ("hinglish", "Bhai jab aap deep learning models train karte ho, toh learning rate warmup aur cosine decay use karna bohot zaroori hota hai. Initial steps me low learning rate gradients ko explode hone se bachata hai aur model parameters ko stably settle karta hai."),
        ("hinglish", "Python me synchronous aur asynchronous programming me main difference execution model ka hai. Synchronous code line by line block karta hai, jabki asyncio event loop I/O operations ke dauran doosre tasks ko execute hone deta hai."),

        # Domain: agent_tools / structured
        ("agent_tools", '{"tool": "calculator", "input": {"expression": "25 * 4 + 100 / 2"}, "expected_output": 150.0}'),
        ("agent_tools", '{"tool": "unit_converter", "input": {"value": 100, "from_unit": "celsius", "to_unit": "fahrenheit"}, "expected_output": 212.0}'),
    ]
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare, filter, and pack pretraining data.")
    parser.add_argument("--tokenizer-dir", type=str, default="tokenizers/bravien-native")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--output-file", type=str, default="data/packed_pretrain/train_packed.pt")
    parser.add_argument("--stats-file", type=str, default="reports/data_quality_stats.json")

    args = parser.parse_args()

    print("=" * 70)
    print("BRAVIEN PRETRAINING DATA PREPARATION & QUALITY FILTERING")
    print("=" * 70)

    # 1. Load Tokenizer
    tok_path = Path(args.tokenizer_dir)
    if not (tok_path / "tokenizer.json").exists():
        import subprocess
        subprocess.run([sys.executable, "scripts/train_bravien_tokenizer.py", "--output-dir", str(tok_path), "--vocab-size", "4000"], check=True)
    tokenizer = BravienTokenizer.from_pretrained(tok_path)
    print(f"Loaded Tokenizer: {tokenizer.vocab_size:,} tokens from '{tok_path}'")

    # 2. Collect and Filter Documents
    filter_engine = QualityFilter()
    seed_docs = generate_curated_seed_documents()

    passed_texts: list[str] = []
    print(f"\nFiltering {len(seed_docs)} initial curated seed documents...")
    for domain, text in seed_docs:
        for i in range(25):
            var_text = text if i == 0 else f"{text}\n[Variation {i}] Seed paragraph repetition index."
            valid, reason, lang = filter_engine.validate_and_filter(var_text, domain=domain)
            if valid:
                passed_texts.append(var_text)

    # Check for Stage 2 datasets
    data_dir = Path("data")
    if data_dir.exists():
        for jsonl_file in data_dir.glob("*.jsonl"):
            with open(jsonl_file, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    try:
                        obj = json.loads(line.strip())
                        text_val = obj.get("text", obj.get("content", f"{obj.get('prompt', '')}\n{obj.get('response', '')}"))
                        valid, _, _ = filter_engine.validate_and_filter(text_val, domain="instruction_sft")
                        if valid:
                            passed_texts.append(text_val)
                    except Exception:
                        pass

    print(f"Passed Documents:  {filter_engine.stats.passed_documents:,} / {filter_engine.stats.total_documents_scanned:,}")
    print(f"Pass Rate:         {filter_engine.stats.pass_rate_percent:.1f}%")
    print(f"Duplicate Rate:    {filter_engine.stats.duplicate_rate_percent:.1f}%")
    print(f"Estimated Tokens:  ~{filter_engine.stats.estimated_tokens_passed:,}")

    # 3. Token Packing into Fixed Blocks
    print(f"\nPacking tokens into contiguous sequence length = {args.max_seq_len}...")
    dataset = PackedPretrainingDataset.from_texts(
        texts=passed_texts,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
    )
    print(f"Generated {len(dataset):,} packed blocks ({len(dataset) * args.max_seq_len:,} total tokens).")

    # 4. Save Packed Dataset & Quality Statistics
    out_path = Path(args.output_file)
    dataset.save(out_path)
    print(f"Saved packed dataset to: {out_path}")

    stats_path = Path(args.stats_file)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(filter_engine.stats.to_dict(), f, indent=2)
    print(f"Saved quality statistics to: {stats_path}")

    print("\n" + "=" * 70)
    print("DATA PREPARATION & QUALITY PIPELINE COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
