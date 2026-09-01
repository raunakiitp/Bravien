"""Test and validate the native Bravien Tokenizer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from bravien.tokenizer.special_tokens import (
    ASSISTANT_CLOSE,
    ASSISTANT_OPEN,
    BOS,
    BOS_ID,
    EOS,
    EOS_ID,
    PAD,
    PAD_ID,
    SPECIAL_TOKENS,
    SYSTEM_CLOSE,
    SYSTEM_OPEN,
    UNK,
    UNK_ID,
    USER_CLOSE,
    USER_OPEN,
)
from bravien.tokenizer.tokenizer import BravienTokenizer

TEST_CASES = [
    # English & Systems
    ("English Prosaic", "Bravien is an independent local assistant with deep reasoning."),
    ("Python Code", "def calculate_loss(pred, target):\n    return torch.mean((pred - target) ** 2)"),
    ("TypeScript Code", "interface TokenizerProps {\n  vocabSize: number;\n  padTokenId: number;\n}"),
    ("Hinglish", "Bhai mujhe yeh code debug karke batao ki off-by-one error kahan hai."),
    ("Hindi (Devanagari)", "कृत्रिम बुद्धिमत्ता और मशीन लर्निंग की तकनीक में निरंतर सुधार हो रहा है।"),
    ("JSON Data", '{"model": "bravien-1.5b", "layers": 32, "status": "active"}'),
    ("Mathematics & LaTeX", "\\int_{0}^{\\infty} e^{-x^2} dx = \\frac{\\sqrt{\\pi}}{2}"),
    ("Special Characters & Emoji", "🚀 Speed: 150 tokens/sec | VRAM: 3.4GB | Accuracy: 99.9% 🎯"),
]


def test_tokenizer(tokenizer_dir: str | Path) -> bool:
    print("\n" + "=" * 70)
    print(f"TESTING BRAVIEN NATIVE TOKENIZER: {tokenizer_dir}")
    print("=" * 70)

    tok_path = Path(tokenizer_dir)
    if not (tok_path / "tokenizer.json").exists() and not tok_path.name.endswith(".json"):
        print(f"Error: Tokenizer file not found in {tok_path}", file=sys.stderr)
        return False

    tokenizer = BravienTokenizer.from_pretrained(tok_path)
    print(f"Vocabulary Size: {tokenizer.vocab_size:,} tokens\n")

    # 1. Special Token ID verification
    print("1. Verifying Special Token IDs:")
    assert tokenizer.pad_token_id == PAD_ID == 0, f"PAD token ID mismatch: {tokenizer.pad_token_id}"
    assert tokenizer.unk_token_id == UNK_ID == 1, f"UNK token ID mismatch: {tokenizer.unk_token_id}"
    assert tokenizer.bos_token_id == BOS_ID == 2, f"BOS token ID mismatch: {tokenizer.bos_token_id}"
    assert tokenizer.eos_token_id == EOS_ID == 3, f"EOS token ID mismatch: {tokenizer.eos_token_id}"
    print(f"   [PASS] Reserved IDs: PAD={PAD_ID}, UNK={UNK_ID}, BOS={BOS_ID}, EOS={EOS_ID}")

    for tok in SPECIAL_TOKENS:
        token_id = tokenizer.token_to_id(tok)
        assert token_id is not None, f"Missing special token {tok}"
        assert tokenizer.id_to_token(token_id) == tok
    print(f"   [PASS] All {len(SPECIAL_TOKENS)} special tokens correctly mapped and bi-directional.")

    # 2. Round-trip and Compression Ratio Testing
    print("\n2. Multilingual & Domain Round-Trip Evaluation:")
    total_chars = 0
    total_bytes = 0
    total_tokens = 0

    for category, text in TEST_CASES:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(encoded, skip_special_tokens=False)

        # Exact lossless reconstruction check
        assert decoded == text, f"Lossy roundtrip for '{category}':\nOriginal: {text}\nDecoded:  {decoded}"

        char_len = len(text)
        byte_len = len(text.encode("utf-8"))
        tok_len = len(encoded)
        bytes_per_tok = byte_len / tok_len if tok_len > 0 else 0

        total_chars += char_len
        total_bytes += byte_len
        total_tokens += tok_len

        print(f"   [{category:<22}] {char_len:>3} chars | {tok_len:>3} tokens | {bytes_per_tok:.2f} bytes/token -> PASS")

    avg_compression = total_bytes / total_tokens if total_tokens > 0 else 0
    print(f"\n   Aggregate Efficiency: {total_bytes} bytes -> {total_tokens} tokens ({avg_compression:.2f} bytes/token)")

    # 3. Chat Template Encoding Test
    print("\n3. Chat Conversation Template Formatting:")
    conversation = [
        {"role": "system", "content": "You are Bravien, an autonomous local assistant."},
        {"role": "user", "content": "Calculate 45 * 12."},
        {"role": "assistant", "content": "45 * 12 = 540."},
    ]
    chat_enc = tokenizer.encode_chat(conversation)
    print(f"   Formatted Chat Tokens: {len(chat_enc.input_ids)}")
    print(f"   Trainable Label Tokens: {chat_enc.num_trainable}")
    assert chat_enc.num_trainable > 0, "Chat encoding should supervise assistant response"
    print("   [PASS] Supervised chat encoding valid.")

    print("\n" + "=" * 70)
    print("ALL TOKENIZER TESTS PASSED (100% LOSSLESS ROUND-TRIP)")
    print("=" * 70 + "\n")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Bravien native tokenizer.")
    parser.add_argument(
        "--tokenizer-dir",
        type=str,
        default="tokenizers/bravien-native",
        help="Directory containing tokenizer.json to evaluate.",
    )
    args = parser.parse_args()

    # Train a baseline tokenizer if not yet present
    if not (Path(args.tokenizer_dir) / "tokenizer.json").exists():
        print(f"Tokenizer not found at {args.tokenizer_dir}. Training an initial seed tokenizer...")
        import subprocess
        subprocess.run([
            sys.executable, "scripts/train_bravien_tokenizer.py",
            "--output-dir", args.tokenizer_dir,
            "--vocab-size", "4000",
        ], check=True)

    success = test_tokenizer(args.tokenizer_dir)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
