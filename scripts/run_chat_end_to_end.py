import json
import urllib.request

prompts = [
    "hi",
    "who are you?",
    "what is 2+2?",
    "my name is Raunak",
    "explain machine learning in one sentence"
]

def test_chat(prompt: str):
    print(f"\n=======================================================")
    print(f"USER: {prompt}")
    data = json.dumps({"messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(
        "http://localhost:3000/api/chat",
        data=data,
        headers={"Content-Type": "application/json"}
    )
    res = urllib.request.urlopen(req)
    deltas = []
    events = []
    meta = None
    complete = None
    for line in res.readlines():
        line_str = line.decode().strip()
        if not line_str or not line_str.startswith("data: "):
            continue
        payload_str = line_str[6:].strip()
        if payload_str == "[DONE]":
            continue
        try:
            payload = json.loads(payload_str)
            kind = payload.get("kind")
            if kind == "content_delta":
                deltas.append(payload.get("delta", ""))
            elif kind == "agent_event":
                events.append(f"{payload.get('eventType')}: {payload.get('message')}")
            elif kind == "meta":
                meta = payload
            elif kind == "message_complete":
                complete = payload
        except Exception:
            pass

    print(f"ROUTING & AGENT EVENTS: {events}")
    print(f"ASSISTANT RESPONSE:\n{''.join(deltas)}")
    if complete:
        timing = complete.get("timing", {})
        usage = complete.get("usage", {})
        print(f"METRICS: {usage.get('outputTokens')} tokens in {timing.get('seconds')}s ({timing.get('tokensPerSecond')} tok/s)")

def main():
    for p in prompts:
        test_chat(p)

if __name__ == "__main__":
    main()
