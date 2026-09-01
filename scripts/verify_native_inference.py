import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import torch
from bravien.inference.native_engine import NativeInferenceEngine

print("Initializing NativeInferenceEngine on GPU...")
engine = NativeInferenceEngine.from_checkpoint("checkpoints/bravien-v4")
print("Engine Info:", engine.info())

print("\nRunning warm-up complete...")
res = engine.complete("Bravien AI Assistant architecture report:", generation={"max_new_tokens": 16, "temperature": 0.0})
print("Warm-up generated:", res.text)
print(f"Prompt tokens: {res.prompt_tokens}, Completion tokens: {res.completion_tokens}, Latency: {res.seconds:.3f}s, T/s: {res.tokens_per_second:.2f}")

print("\nTesting chat completion with message formatting...")
chat_prompt, plan = engine.build_chat_prompt_within_context([
    {"role": "user", "content": "What is the capital of France?"}
], max_new_tokens=16)
print("Rendered Chat Prompt:\n", chat_prompt)
chat_res = engine.complete(chat_prompt, generation={"max_new_tokens": 16, "temperature": 0.0})
print("Chat Output:", chat_res.text)

print("\nSUCCESS: Native inference engine test completed successfully!")
