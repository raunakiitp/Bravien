"""Exercise the local API in-process with FastAPI's TestClient."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient

from bravien.inference.engine import EngineConfig
from bravien.inference.server import MAX_BODY_BYTES, create_app
from bravien.model.generation import GenerationConfig
from bravien.utils.logging import configure_stdout, setup_logging

configure_stdout()
setup_logging(level="WARNING")

RUN_DIR = REPO_ROOT / "checkpoints" / "smoke-pretrain"

app = create_app(
    checkpoint=RUN_DIR,
    engine_config=EngineConfig(
        default_generation=GenerationConfig(
            max_new_tokens=40, temperature=0.8, top_k=40, seed=0
        )
    ),
)
client = TestClient(app)

print("=== GET /health ===")
r = client.get("/health")
print(r.status_code, json.dumps(r.json(), indent=2))

print("\n=== GET /v1/models ===")
r = client.get("/v1/models")
data = r.json()["data"][0]
print(r.status_code, json.dumps({k: data[k] for k in
      ("name", "parameters", "layers", "context_length", "vocab_size", "precision")}, indent=2))
print("training block:", json.dumps(data["training"], indent=2, default=str)[:400])

print("\n=== GET /v1/models/{name} ===")
print(client.get(f"/v1/models/{data['name']}").status_code)
print(client.get("/v1/models/gpt-4o").status_code, client.get("/v1/models/gpt-4o").json())

print("\n=== POST /v1/completions (non-streaming) ===")
r = client.post("/v1/completions", json={"prompt": "Notes on maps.\n\nThe pilot", "max_tokens": 25, "seed": 7})
print(r.status_code, json.dumps(r.json(), indent=2))

print("\n=== POST /v1/completions (SSE stream) ===")
with client.stream("POST", "/v1/completions",
                   json={"prompt": "Notes on maps.\n\nThe pilot", "max_tokens": 25,
                         "seed": 7, "stream": True}) as resp:
    print("status", resp.status_code, "content-type", resp.headers.get("content-type"))
    frames = []
    for line in resp.iter_lines():
        if line.startswith("data: "):
            frames.append(line[6:])
print(f"frames: {len(frames)}")
deltas = [json.loads(f)["delta"] for f in frames if f != "[DONE]" and json.loads(f)["kind"] == "content_delta"]
print("streamed text:", repr("".join(deltas)))
print("last two frames:", frames[-2], "|", frames[-1])
non_stream_text = r.json()["text"]
print(f"stream matches non-stream (same seed): {''.join(deltas) == non_stream_text}")

print("\n=== POST /v1/chat/completions (SSE) ===")
with client.stream("POST", "/v1/chat/completions",
                   json={"messages": [{"role": "user", "content": "Tell me about clocks."}],
                         "max_tokens": 20, "seed": 3, "stream": True}) as resp:
    chat_frames = [l[6:] for l in resp.iter_lines() if l.startswith("data: ")]
print("text:", repr("".join(json.loads(f)["delta"] for f in chat_frames
                           if f != "[DONE]" and json.loads(f)["kind"] == "content_delta")))
print("complete frame:", [f for f in chat_frames if f != "[DONE]" and json.loads(f)["kind"] == "message_complete"])

print("\n=== POST /v1/score ===")
r = client.post("/v1/score", json={"text": "The engineer measured the tide charts."})
print(r.status_code, r.json())

print("\n=== validation ===")
cases = [
    ("empty prompt", "/v1/completions", {"prompt": ""}),
    ("unknown field", "/v1/completions", {"prompt": "x", "frequency_penalty": 1}),
    ("temperature too high", "/v1/completions", {"prompt": "x", "temperature": 99}),
    ("max_tokens too high", "/v1/completions", {"prompt": "x", "max_tokens": 999999}),
    ("bad role", "/v1/chat/completions", {"messages": [{"role": "root", "content": "x"}]}),
    ("no messages", "/v1/chat/completions", {"messages": []}),
    ("too many stops", "/v1/completions", {"prompt": "x", "stop": ["a"] * 20}),
    ("long stop string", "/v1/completions", {"prompt": "x", "stop": ["z" * 100]}),
]
for label, path, body in cases:
    resp = client.post(path, json=body)
    print(f"  {label}: {resp.status_code}")

print("\n=== body size limit ===")
big = "a" * (MAX_BODY_BYTES + 100)
r = client.post("/v1/completions", json={"prompt": big})
print(f"  {len(big):,} char prompt -> {r.status_code} {r.json().get('error', {}).get('code')}")

print("\n=== CORS ===")
r = client.options("/v1/completions", headers={
    "Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"})
print(f"  localhost:3000 -> {r.status_code} allow-origin={r.headers.get('access-control-allow-origin')}")
r = client.options("/v1/completions", headers={
    "Origin": "https://evil.example.com", "Access-Control-Request-Method": "POST"})
print(f"  evil.example.com -> {r.status_code} allow-origin={r.headers.get('access-control-allow-origin')}")

print("\n=== no-model server reports why ===")
empty_app = create_app(checkpoint=REPO_ROOT / "checkpoints" / "does-not-exist")
empty = TestClient(empty_app)
h = empty.get("/health").json()
print("  /health:", json.dumps({k: h[k] for k in ("status", "model_loaded", "error")}, indent=2)[:300])
r = empty.post("/v1/completions", json={"prompt": "hello"})
print("  /v1/completions:", r.status_code, r.json())
print("  /v1/models:", empty.get("/v1/models").json())
