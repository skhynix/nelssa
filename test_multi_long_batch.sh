#!/bin/bash

# Multi-Long Batch Test for P/D Disaggregated (NELSSA)
# Sends N concurrent long requests of identical length simultaneously and
# reports per-request streaming latency (TTFT, decode TPOT, throughput) plus
# the aggregated batch throughput (total tokens / wall time) — the metric that
# shows whether NELSSA batches multiple long requests together.
#
# Requires the P/D stack up (run_pd_with_nelssa.sh) with:
#   --nelssa-max-num-long-requests >= N   (else only 1 long request is served
#   at a time; the rest queue, so you won't see batch speedup)
#
# Usage:
#   NREQS=3 TOKENS=96000 MAX_TOKENS=200 bash test_multi_long_batch.sh
#
# Env:
#   NREQS       number of concurrent requests (default 2)
#   TOKENS      prompt length in words (default 96000; ~99.8K actual tokens)
#   MAX_TOKENS  output tokens per request (default 200)
#   MODEL       model path (defaults to the Llama-3.1-8B-Instruct snapshot)
#   PROXY_URL   proxy endpoint (default http://localhost:8300/v1/completions)

set -e

MODEL="${MODEL:-/mnt/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659}"
PROXY_URL="${PROXY_URL:-http://localhost:8300/v1/completions}"
NREQS="${NREQS:-2}"
TOKENS="${TOKENS:-96000}"
MAX_TOKENS="${MAX_TOKENS:-200}"

REQ_DIR="${REQ_DIR:-$(mktemp -d -p . multi_long_batch.XXXXXX)}"

echo "======================================"
echo "Multi-Long Batch Test"
echo "======================================"
echo "Concurrent requests : $NREQS"
echo "Prompt (words)      : $TOKENS"
echo "Output tokens/req   : $MAX_TOKENS"
echo "Proxy               : $PROXY_URL"
echo "Temp dir            : $REQ_DIR"
echo "======================================"
echo ""

# Build one shared request body (identical length for all requests so the
# per-request workload is the same; differences come purely from batching).
python3 - "$MODEL" "$TOKENS" "$MAX_TOKENS" "$REQ_DIR/body.json" <<'PY'
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
print(f"Shared prompt chars: {len(text)} -> {out_path}")
PY

# Send N concurrent streaming requests, one Python process per request so they
# run in true parallel (urllib is blocking). Each writes a per-request JSON
# result; the wall-clock window is measured across all of them.
python3 - "$NREQS" "$REQ_DIR" "$PROXY_URL" <<'PY'
import json, os, sys, time, threading, urllib.request

nreqs = int(sys.argv[1])
req_dir = sys.argv[2]
proxy = sys.argv[3]
with open(os.path.join(req_dir, "body.json")) as f:
    body = json.load(f)

results = [None] * nreqs

def run_one(i):
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
                if not line or line.startswith(":"):
                    continue
                if line.startswith("data:"):
                    line = line[len("data:"):].strip()
                if line == "[DONE]":
                    t_done = time.perf_counter()
                    break
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
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
        t_done = t_done or time.perf_counter()
    except Exception as e:
        results[i] = {"error": str(e)}
        return
    n = len(tokens)
    results[i] = {
        "n": n,
        "ttft": (t_first - t_send) * 1e3 if t_first else None,
        "gen_ms": (t_last - t_first) * 1e3 if (t_first and t_last) else 0.0,
        "e2e_ms": (t_done - t_send) * 1e3,
        "decode_tpot": ((t_last - t_first) / (n - 1) * 1e3) if (t_first and t_last and n > 1) else 0.0,
        "t_send": t_send, "t_first": t_first, "t_last": t_last, "t_done": t_done,
    }

threads = [threading.Thread(target=run_one, args=(i,), daemon=True) for i in range(nreqs)]
t_wall_start = time.perf_counter()
for t in threads:
    t.start()
for t in threads:
    t.join()
t_wall_end = time.perf_counter()

# Per-request report
print()
print("======================================")
print(f"Per-request latency (streaming, prefill/decode split) — {nreqs} concurrent")
print("======================================")
total_tokens = 0
ok = 0
for i, r in enumerate(results):
    if not r or "error" in r:
        print(f"  req{i}: FAILED ({r.get('error') if r else 'no result'})")
        continue
    total_tokens += r["n"]
    ok += 1
    tp = 1000.0 / r["decode_tpot"] if r["decode_tpot"] else 0.0
    print(f"  req{i}: out={r['n']:3d}  TTFT={r['ttft']:8.1f}ms  "
          f"decode TPOT={r['decode_tpot']:7.1f} ms/tok  "
          f"throughput={tp:6.2f} tok/s  e2e={r['e2e_ms']:8.1f}ms")

# Aggregated batch throughput
wall_ms = (t_wall_end - t_wall_start) * 1e3
agg_tp = (total_tokens / (wall_ms / 1e3)) if wall_ms > 0 else 0.0
# Sum of per-request throughputs (no batching → this ≈ agg_tp; with batching,
# agg_tp exceeds the sum of single-request rates because decode work overlaps).
sum_tp = sum(1000.0 / r["decode_tpot"] for r in results if r and "error" not in r and r["decode_tpot"])
print()
print("======================================")
print("Aggregated (batch) throughput")
print("======================================")
print(f"  requests completed    : {ok}/{nreqs}")
print(f"  total output tokens   : {total_tokens}")
print(f"  wall time             : {wall_ms:.1f} ms")
print(f"  agg throughput        : {agg_tp:.2f} tok/s  (total tokens / wall time)")
print(f"  sum per-req throughput: {sum_tp:.2f} tok/s  (upper bound if NO batching/overlap)")
if sum_tp > 0:
    print(f"  batch overlap ratio   : {agg_tp / sum_tp:.2f}x  (>1 means decode work overlapped)")
print("======================================")

# Persist results for later comparison
with open(os.path.join(req_dir, "results.json"), "w") as f:
    json.dump({"nreqs": nreqs, "results": results,
               "wall_ms": wall_ms, "total_tokens": total_tokens,
               "agg_tp": agg_tp, "sum_tp": sum_tp}, f, indent=2)
print(f"\nResults saved to {req_dir}/results.json")
PY

echo ""
echo "======================================"
echo "Test complete"
echo "======================================"