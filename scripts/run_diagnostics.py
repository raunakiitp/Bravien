import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bravien.inference.hf_engine import HFInferenceEngine

def main():
    print("Loading Qwen/Qwen2.5-0.5B-Instruct...")
    engine = HFInferenceEngine.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    print(f"Loaded: {engine.model_name} on {engine.device_info.name}")

    prompts = [
        "hi",
        "who are you?",
        "what is 2+2?",
        "my name is Raunak",
        "explain machine learning in one sentence"
    ]

    for p in prompts:
        chat_prompt = engine.build_chat_prompt([{"role": "user", "content": p}])
        print(f"\n==========================================")
        print(f"USER: {p}")
        print(f"RAW CHAT PROMPT:\n{repr(chat_prompt)}")
        res = engine.complete(chat_prompt)
        print(f"RESPONSE:\n{res.text}")
        print(f"SPEED: {res.tokens_per_second} tok/s ({res.completion_tokens} tokens in {res.seconds}s)")

if __name__ == "__main__":
    main()
