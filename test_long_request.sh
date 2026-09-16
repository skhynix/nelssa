#!/bin/bash

# Long Request Test for P/D Disaggregated
# Tests with a long prompt (~96K tokens by default; override via $TOKENS).

MODEL="/mnt/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
PROXY_URL="http://localhost:8300/v1/completions"
TARGET_TOKENS="${TOKENS:-100000}"
MAX_TOKENS="${MAX_TOKENS:-200}"
REQ_JSON="/tmp/long_req_body.json"
OUT_JSON="/tmp/long_req_out.json"

echo "======================================"
echo "Long Request Test (${TARGET_TOKENS} tokens)"
echo "======================================"

# Build the request JSON in a file (via python, not shell interpolation) so the
# prompt can be arbitrarily long. Passing a huge prompt as a curl CLI argument
# overflows ARG_MAX ("Argument list too long"); `curl -d @file` avoids that.
python3 - "$MODEL" "$TARGET_TOKENS" "$MAX_TOKENS" "$REQ_JSON" <<'PY'
import json, random, sys

model, target_tokens, max_tokens, out_path = sys.argv[1:5]
target_tokens = int(target_tokens)
max_tokens = int(max_tokens)
random.seed(42)

paragraph = (
    'Space exploration has expanded the horizons of human knowledge in '
    'ways that were unimaginable only a century ago. Engineers and scientists '
    'around the world collaborate to design rockets, satellites, and robotic '
    'landers that travel far beyond Earth. Each mission gathers data about '
    'distant planets, their moons, and the faint light of stars that may host '
    'other worlds. The challenges are immense: extreme temperatures, intense '
    'radiation, and the vast distances that even light takes years to cross. '
    'Yet every successful launch teaches us something new about the universe '
    'and our place within it. Robotic rovers photograph alien landscapes, '
    'telescopes capture the earliest light from the dawn of time, and crews '
    'aboard orbiting stations run experiments in microgravity. Together these '
    'efforts slowly build a map of the cosmos and a vision of what humanity '
    'might one day reach. The story of space is also the story of people who '
    'asked bold questions and built machines to answer them.'
)

sentences = [s.strip() for s in paragraph.split('.') if s.strip()]
words = []
while len(words) < target_tokens:
    order = sentences[:]
    random.shuffle(order)
    for s in order:
        words.extend(s.split())
        words.append('.')
        if len(words) >= target_tokens:
            break
text = ' '.join(words[:target_tokens])

body = {"model": model, "prompt": text,
        "max_tokens": max_tokens, "temperature": 0.0, "stream": True}
with open(out_path, "w") as f:
    json.dump(body, f)
print(f"Prompt chars: {len(text)} -> {out_path}")
PY

echo "Sending long request (body: $(wc -c < "$REQ_JSON") bytes, streaming)..."

# Streaming: split prefill (TTFT) from decode (inter-token).
# Events (ms since process start):
#   t_send      : request sent
#   t_first     : first output token received  -> TTFT (time-to-first-token)
#   t_last      : last output token received
# Decode-only TPOT = (t_last - t_first) / (n_tokens - 1)  -- excludes prefill.
REQ_JSON=$REQ_JSON OUT_JSON=$OUT_JSON PROXY_URL=$PROXY_URL python3 - "$OUT_JSON" <<'PY'
import json, os, time, sys, urllib.request

out_path = sys.argv[1]
proxy = os.environ['PROXY_URL']
req_path = os.environ['REQ_JSON']
with open(req_path) as f:
    body = json.load(f)

req = urllib.request.Request(
    proxy, data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"})
t_send = time.perf_counter()
tokens = []
t_first = None
t_last = None
t_done = None

try:
    with urllib.request.urlopen(req) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            if line.startswith(":"):
                continue  # SSE comment / heartbeat
            if line.startswith("data:"):
                line = line[len("data:"):].strip()
            if line == "[DONE]":
                t_done = time.perf_counter()
                break
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            # OpenAI-style streaming chunk
            if ev.get("object") == "chat.completion.chunk" or "choices" in ev:
                ch = ev.get("choices", [{}])[0]
                delta = ch.get("delta", {})
                tok = delta.get("content") or ch.get("text") or ""
            else:
                tok = ""
            if tok:
                now = time.perf_counter()
                if t_first is None:
                    t_first = now
                tokens.append(tok)
                t_last = now
            # usage sometimes in final chunk
            if ev.get("usage"):
                usage = ev["usage"]
        t_done = t_done or time.perf_counter()
except Exception as e:
    print(f"Request failed: {e}", file=sys.stderr)
    sys.exit(1)

n = len(tokens)
prompt_tokens = body["prompt"].count(" ") + 1  # rough; server returns exact in usage

print()
print("======================================")
print("Latency metrics (streaming, prefill/decode split)")
print("======================================")
print(f"Prompt tokens:       {prompt_tokens} (approx)")
print(f"Output tokens:       {n}")
if n > 0:
    ttft_ms = (t_first - t_send) * 1000
    gen_ms = (t_last - t_first) * 1000
    e2e_ms = (t_done - t_send) * 1000
    # Decode-only TPOT excludes prefill (TTFT).
    if n > 1:
        decode_tpot = (t_last - t_first) / (n - 1) * 1000
        decode_tp = 1000.0 / decode_tpot
    else:
        decode_tpot = gen_ms
        decode_tp = 0.0
    # E2E TPOT (old-style, includes prefill) for comparison.
    e2e_tpot = e2e_ms / n if n else 0.0
    print(f"TTFT:                {ttft_ms:.1f} ms   (prefill: request -> first token)")
    print(f"Generation time:     {gen_ms:.1f} ms   (first -> last token)")
    print(f"E2E latency:         {e2e_ms:.1f} ms   (request -> done)")
    print(f"Decode TPOT:         {decode_tpot:.1f} ms/token   (prefill EXCLUDED)")
    print(f"Decode throughput:   {decode_tp:.2f} tokens/s")
    print(f"E2E TPOT (old):      {e2e_tpot:.1f} ms/token   (prefill INCLUDED)")
print()
print("Output text:")
print(repr("".join(tokens)))
PY

echo ""
echo "======================================"
echo "Test complete"
echo "======================================"