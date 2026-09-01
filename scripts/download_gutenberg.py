"""Download many Project Gutenberg books for pretraining.

Downloads ~50-100 popular English texts from Gutenberg and saves them
as .txt files in datasets/raw/ so prepare_data.py can tokenize them.

    python scripts/download_gutenberg.py
    python scripts/download_gutenberg.py --output-dir datasets/raw --count 80
"""
from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Popular Gutenberg IDs with filenames
BOOKS = [
    # Already have these
    (11, "alice-in-wonderland"),
    (84, "frankenstein"),
    (1342, "pride-and-prejudice"),
    (1661, "sherlock-holmes"),
    # More classics
    (98, "tale-of-two-cities"),
    (1232, "the-prince-machiavelli"),
    (174, "picture-of-dorian-gray"),
    (2701, "moby-dick"),
    (345, "dracula"),
    (1400, "great-expectations"),
    (76, "huckleberry-finn"),
    (74, "tom-sawyer"),
    (25344, "the-scarlet-letter"),
    (2554, "crime-and-punishment"),
    (28054, "brothers-karamazov"),
    (2600, "war-and-peace"),
    (1260, "jane-eyre"),
    (161, "sense-and-sensibility"),
    (158, "emma"),
    (141, "mansfield-park"),
    (105, "persuasion"),
    (42, "adventures-of-sherlock-holmes"),
    (2852, "hound-of-baskervilles"),
    (863, "the-mysterious-island"),
    (103, "around-the-world-80-days"),
    (164, "twenty-thousand-leagues"),
    (3207, "journey-to-center"),
    (5230, "time-machine"),
    (35, "war-of-the-worlds"),
    (36, "invisible-man"),
    (1080, "modest-proposal"),
    (768, "wuthering-heights"),
    (1952, "yellow-wallpaper"),
    (514, "little-women"),
    (55, "wizard-of-oz"),
    (16, "peter-pan"),
    (219, "heart-of-darkness"),
    (730, "oliver-twist"),
    (766, "david-copperfield"),
    (580, "dombey-and-son"),
    (786, "bleak-house"),
    (1023, "bleak-house-2"),
    (46, "christmas-carol"),
    (1400, "great-expectations-alt"),
    (2097, "meditations-aurelius"),
    (4280, "nicomachean-ethics"),
    (1497, "republic-plato"),
    (1998, "apology-plato"),
    (1321, "leviathan-hobbes"),
    (10615, "wealth-of-nations"),
    (7370, "common-sense"),
    (3207, "social-contract"),
    (2650, "thus-spoke-zarathustra"),
    (5827, "beyond-good-evil"),
    (4280, "ethics-aristotle"),
    (996, "don-quixote"),
    (2413, "anna-karenina"),
    (2680, "meditations"),
    (14833, "iliad"),
    (3160, "odyssey"),
    (21279, "aeneid"),
    (1727, "beowulf"),
    (100, "complete-shakespeare"),
    (1000, "canterbury-tales"),
    (8800, "paradise-lost"),
    (120, "treasure-island"),
    (215, "call-of-the-wild"),
    (1228, "sea-wolf"),
    (23, "northanger-abbey"),
    (244, "mysterious-affair"),
    (863, "swiss-family-robinson"),
    (236, "jungle-book"),
    (35, "war-worlds-2"),
    (6761, "arabian-nights"),
    (4300, "ulysses"),
    (2814, "dubliners"),
    (29809, "portrait-artist"),
    (3825, "sun-also-rises"),
    (182, "farewell-to-arms"),
    (1184, "count-monte-cristo"),
    (2527, "three-musketeers"),
    (62, "miserables"),
    (135, "notre-dame"),
    (2800, "dune-homage"),
    (41445, "flatland"),
    (16328, "narrative-frederick-douglass"),
    (1322, "uncle-toms-cabin"),
    (203, "walden"),
    (243, "civil-disobedience"),
    (1250, "confessions-augustine"),
    (3207, "social-contract-2"),
    (4705, "wealth-nations-2"),
]

GUTENBERG_MIRRORS = [
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt",
    "https://www.gutenberg.org/files/{id}/{id}-0.txt",
    "https://www.gutenberg.org/files/{id}/{id}.txt",
]


def download_book(book_id: int, name: str, output_dir: Path) -> bool:
    """Download one Gutenberg book. Returns True if successful."""
    # Skip if any version already exists
    existing = list(output_dir.glob(f"*{book_id}*")) + list(output_dir.glob(f"*{name}*"))
    if existing:
        print(f"  [skip] {name} ({book_id}) — already present")
        return True

    out_path = output_dir / f"gutenberg-{name}-{book_id}.txt"
    for url_template in GUTENBERG_MIRRORS:
        url = url_template.format(id=book_id)
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (academic; gutenberg-downloader/1.0)"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                content = resp.read()
            # Must be at least 10KB to be a real book
            if len(content) < 10_000:
                continue
            # Try to decode
            for enc in ("utf-8", "latin-1"):
                try:
                    text = content.decode(enc)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                continue
            out_path.write_text(text, encoding="utf-8")
            size_kb = len(content) // 1024
            print(f"  [ok]   {name} ({book_id}) — {size_kb} KB")
            return True
        except Exception:
            continue
    print(f"  [fail] {name} ({book_id}) — all mirrors failed")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Gutenberg books for pretraining.")
    parser.add_argument("--output-dir", default="datasets/raw")
    parser.add_argument("--count", type=int, default=len(BOOKS), help="Max books to download")
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds between requests")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Deduplicate by book_id
    seen_ids: set[int] = set()
    unique_books = []
    for book_id, name in BOOKS:
        if book_id not in seen_ids:
            seen_ids.add(book_id)
            unique_books.append((book_id, name))

    books_to_download = unique_books[: args.count]
    print(f"Downloading {len(books_to_download)} books to {output_dir}/")
    print()

    ok = fail = skip = 0
    for book_id, name in books_to_download:
        result = download_book(book_id, name, output_dir)
        if result:
            ok += 1
        else:
            fail += 1
        time.sleep(args.delay)

    print()
    print(f"Done: {ok} downloaded, {fail} failed")
    print()

    # Show total size
    txts = list(output_dir.glob("*.txt"))
    total_mb = sum(f.stat().st_size for f in txts) / (1024 * 1024)
    print(f"Total .txt files: {len(txts)} ({total_mb:.1f} MB)")
    print()
    print("Next: python scripts/prepare_data.py --local-dir datasets/raw --name big-corpus")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
