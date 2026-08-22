"""Manual check of the inference engine against the trained smoke checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import json

from bravien.inference.engine import EngineConfig, InferenceEngine, StreamDecoder
from bravien.model.generation import GenerationConfig
from bravien.utils.logging import configure_stdout, setup_logging

configure_stdout()
setup_logging(level="INFO")

RUN_DIR = REPO_ROOT / "checkpoints" / "smoke-pretrain"

engine = InferenceEngine.from_checkpoint(
    RUN_DIR,
    config=EngineConfig(
        default_generation=GenerationConfig(
            max_new_tokens=60, temperature=0.8, top_k=40, top_p=0.95,
            repetition_penalty=1.05, seed=0,
        )
    ),
)

print("\n=== engine.info() ===")
print(json.dumps(engine.info(), indent=2, default=str))

prompt = "Notes on tides.\n\nThe engineer measured"

print("\n=== streaming (deltas shown with | separators) ===")
pieces = []
for event in engine.stream(prompt):
    if event.done:
        print(f"\n\nfinish={event.finish_reason} usage={ {k: v for k, v in event.usage.items() if k != 'text'} }")
        streamed_text = event.usage["text"]
    else:
        pieces.append(event.text)
        print(event.text, end="|", flush=True)

joined = "".join(pieces)
print(f"\nstreamed deltas == final text: {joined == streamed_text}")

print("\n=== non-streaming complete() ===")
result = engine.complete(prompt)
print(json.dumps(result.to_dict(), indent=2))

print("\n=== determinism: same seed twice ===")
a = engine.complete(prompt, generation={"seed": 123})
b = engine.complete(prompt, generation={"seed": 123})
print(f"identical: {a.text == b.text}")
c = engine.complete(prompt, generation={"seed": 456})
print(f"different seed differs: {a.text != c.text}")

print("\n=== greedy is deterministic without a seed ===")
g1 = engine.complete(prompt, generation={"temperature": 0.0, "seed": None})
g2 = engine.complete(prompt, generation={"temperature": 0.0, "seed": None})
print(f"identical: {g1.text == g2.text}")
print(f"greedy text: {g1.text[:120]!r}")

print("\n=== chat template path ===")
messages = [
    {"role": "system", "content": "You are Bravien."},
    {"role": "user", "content": "Describe the harbour lights."},
]
print("prompt:", repr(engine.build_chat_prompt(messages))[:300])
chat = engine.chat(messages, generation={"max_new_tokens": 30})
print(json.dumps(chat.to_dict(), indent=2))

print("\n=== stop strings ===")
stopped = engine.complete(prompt, generation={"seed": 0}, stop_strings=["."])
print(f"finish={stopped.finish_reason} text={stopped.text!r}")
assert "." not in stopped.text, "stop string leaked into output"

print("\n=== streamed stop string never leaks ===")
acc = ""
for event in engine.stream(prompt, generation={"seed": 0}, stop_strings=["."]):
    if not event.done:
        acc += event.text
print(f"accumulated={acc!r}")
assert "." not in acc, "stop string leaked into a stream delta"

print("\n=== limits and validation ===")
for label, fn in [
    ("empty prompt", lambda: engine.complete("")),
    ("bad param", lambda: engine.complete("x", generation={"nope": 1})),
    ("huge prompt", lambda: engine.complete("a" * 300_000)),
    ("empty messages", lambda: engine.chat([])),
    ("bad message", lambda: engine.chat([{"role": "user"}])),
]:
    try:
        fn()
        print(f"  {label}: NO ERROR (unexpected)")
    except Exception as exc:
        print(f"  {label}: {type(exc).__name__}: {str(exc)[:80]}")

print("\n=== max_new_tokens is clamped ===")
engine.config.max_new_tokens_limit = 5
short = engine.complete(prompt, generation={"max_new_tokens": 10_000})
print(f"asked 10000, got {short.completion_tokens} tokens")
engine.config.max_new_tokens_limit = 2048

print("\n=== StreamDecoder handles multi-byte characters ===")
tok = engine.tokenizer
text = "café — naïve 日本語 🙂"
ids = tok.encode(text)
dec = StreamDecoder(tok, skip_special=False)
out = "".join(dec.push(i) for i in ids) + dec.flush()
print(f"  input:  {text!r}")
print(f"  output: {out!r}")
print(f"  incremental == batch decode: {out == tok.decode(ids, skip_special_tokens=False)}")

print("\n=== logprob scoring ===")
print(json.dumps(engine.logprob("The engineer measured the tide charts in the observatory."), indent=2))
print(json.dumps(engine.logprob("zzz qqq xxx vvv www"), indent=2))
