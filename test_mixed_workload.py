#!/usr/bin/env python3
"""Mixed Workload benchmark for P/D disaggregated NELSSA.

Short requests are sent at a fixed QPS (a background stream) while long requests
are injected periodically. The point is to see whether the short requests'
TPOT / p99 degrade while a long request occupies the GPU KV cache — the vLLM
baseline should degrade (KV-cache pressure from the long request), and NELSSA
(which offloads the long request's KV to CPU) should keep the short requests fast.

HTTP client: talks to the proxy of a running run_pd_with_nelssa.sh stack
(NELSSA enabled, or NELSSA_OFF=1 for the baseline). Reuses the streaming
TTFT/TPOT measurement pattern from test_long_request.sh.

Usage (typically via the test_mixed_workload.sh wrapper):
    python3 test_mixed_workload.py \
        --short-qps 2 --short-tokens 500 \
        --long-tokens 96000 --long-interval 10 --long-count 3 \
        --short-max-tokens 50 --long-max-tokens 200 \
        --model <path> --proxy http://localhost:8300/v1/completions
"""

import argparse
import json
import random
import threading
import time
import urllib.request


def build_prompt(target_tokens, seed=42):
    """Build a prompt of ~target_tokens tokens (words), matching the
    test_long_request.sh paragraph generator so prompt shapes are consistent."""
    random.seed(seed)
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
    return ' '.join(words[:target_tokens])


def stream_request(proxy, body, results, idx, kind, inject_time):
    """Send one streaming request; record TTFT / decode TPOT into results[idx]."""
    req = urllib.request.Request(
        proxy, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t_send = time.perf_counter()
    t_first = None
    t_last = None
    t_done = None
    n = 0
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
                    n += 1
                    t_last = now
        t_done = t_done or time.perf_counter()
    except Exception as e:
        results[idx] = {"kind": kind, "error": str(e),
                        "inject_time": inject_time, "t_send": t_send}
        return
    decode_tpot = ((t_last - t_first) / (n - 1) * 1e3) if (t_first and t_last and n > 1) else 0.0
    results[idx] = {
        "kind": kind,
        "n_tokens": n,
        "ttft_ms": (t_first - t_send) * 1e3 if t_first else None,
        "decode_tpot_ms": decode_tpot,
        "decode_tp": 1000.0 / decode_tpot if decode_tpot else 0.0,
        "e2e_ms": (t_done - t_send) * 1e3,
        "t_send": t_send,
    }


def main():
    p = argparse.ArgumentParser(description="Mixed Workload benchmark (short @ QPS + periodic long)")
    p.add_argument("--short-qps", type=float, default=2.0)
    p.add_argument("--short-tokens", type=int, default=500)
    p.add_argument("--long-tokens", type=int, default=96000)
    p.add_argument("--long-interval", type=float, default=10.0)
    p.add_argument("--long-count", type=int, default=3)
    p.add_argument("--short-max-tokens", type=int, default=50)
    p.add_argument("--long-max-tokens", type=int, default=200)
    p.add_argument("--duration", type=float, default=None,
                   help="total wall-time cap (s); default = long_count*interval + tail")
    p.add_argument("--model", type=str,
                   default="/mnt/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659")
    p.add_argument("--proxy", type=str, default="http://localhost:8300/v1/completions")
    p.add_argument("--out", type=str, default="mixed_workload_results.json")
    args = p.parse_args()

    if args.duration is None:
        args.duration = args.long_count * args.long_interval + 30.0

    short_body = {"model": args.model, "prompt": build_prompt(args.short_tokens, seed=7),
                  "max_tokens": args.short_max_tokens, "temperature": 0.0, "stream": True}
    long_body = {"model": args.model, "prompt": build_prompt(args.long_tokens, seed=42),
                 "max_tokens": args.long_max_tokens, "temperature": 0.0, "stream": True}

    print("=" * 70)
    print(f"Mixed Workload: short={args.short_tokens}tok @ {args.short_qps} QPS, "
          f"long={args.long_tokens}tok x{args.long_count} every {args.long_interval}s")
    print(f"Duration cap: {args.duration:.0f}s  Proxy: {args.proxy}")
    print("=" * 70)

    results = {}
    threads = []
    t_start = time.perf_counter()

    def add(kind, body, inject_time):
        idx = f"{kind}_{len(results)}"
        results[idx] = None
        th = threading.Thread(target=stream_request,
                              args=(args.proxy, body, results, idx, kind, inject_time),
                              daemon=True)
        th.start()
        threads.append(th)
        return th

    def long_loop():
        for k in range(args.long_count):
            t_fire = k * args.long_interval
            delay = t_fire - (time.perf_counter() - t_start)
            if delay > 0:
                time.sleep(delay)
            add("long", long_body, time.perf_counter() - t_start)
            print(f"[long] injected #{k+1}/{args.long_count} at t={time.perf_counter()-t_start:.1f}s")

    lt = threading.Thread(target=long_loop, daemon=True)
    lt.start()

    short_period = 1.0 / args.short_qps if args.short_qps > 0 else 1.0
    while time.perf_counter() - t_start < args.duration:
        add("short", short_body, time.perf_counter() - t_start)
        time.sleep(short_period)

    for th in threads:
        th.join(timeout=300.0)

    shorts = [r for r in results.values() if r and r.get("kind") == "short" and "error" not in r]
    longs = [r for r in results.values() if r and r.get("kind") == "long" and "error" not in r]
    errors = [r for r in results.values() if r and "error" in r]

    def stats(name, vals):
        if not vals:
            print(f"  {name}: n=0")
            return None
        vs = sorted(vals)
        n = len(vs)
        med = vs[n // 2]
        p99 = vs[min(n - 1, int(n * 0.99) - (1 if n > 1 else 0))]
        mean = sum(vs) / n
        print(f"  {name}: n={n}  mean={mean:.1f}  med={med:.1f}  p99={p99:.1f}  "
              f"min={vs[0]:.1f}  max={vs[-1]:.1f}")
        return {"mean": mean, "med": med, "p99": p99, "min": vs[0], "max": vs[-1]}

    print()
    print("=" * 70)
    print(f"Short requests (n={len(shorts)})")
    print("=" * 70)
    short_tpots = [r["decode_tpot_ms"] for r in shorts if r["decode_tpot_ms"] > 0]
    short_ttfts = [r["ttft_ms"] for r in shorts if r["ttft_ms"] is not None]
    s_tpot = stats("short decode TPOT (ms/tok)", short_tpots)
    s_ttft = stats("short TTFT (ms)", short_ttfts)

    print()
    print("Short TPOT around long injections (degradation check):")
    long_send_times = sorted(r["t_send"] - t_start for r in longs)
    for li, lt_fire in enumerate(long_send_times):
        window = [r["decode_tpot_ms"] for r in shorts
                  if r["decode_tpot_ms"] > 0
                  and lt_fire - 2 <= (r["t_send"] - t_start) <= lt_fire + 15]
        before = [r["decode_tpot_ms"] for r in shorts
                  if r["decode_tpot_ms"] > 0
                  and lt_fire - 12 <= (r["t_send"] - t_start) < lt_fire - 2]
        if window:
            med_w = sorted(window)[len(window) // 2]
            med_b = (sorted(before)[len(before) // 2] if before else 0.0) or med_w
            print(f"  long #{li+1} @ t={lt_fire:.1f}s: "
                  f"before med={med_b:.1f}  during/after med={med_w:.1f}  "
                  f"ratio={med_w/med_b:.2f}x")

    print()
    print("=" * 70)
    print(f"Long requests (n={len(longs)})")
    print("=" * 70)
    l_tpot = stats("long decode TPOT (ms/tok)",
                   [r["decode_tpot_ms"] for r in longs if r["decode_tpot_ms"] > 0])
    l_ttft = stats("long TTFT (ms)", [r["ttft_ms"] for r in longs if r["ttft_ms"] is not None])

    print()
    print(f"Errors: {len(errors)}")
    for e in errors[:5]:
        print(f"  {e['kind']}: {e.get('error')}")

    summary = {
        "config": vars(args),
        "short": {"n": len(shorts), "tpot_ms": s_tpot, "ttft_ms": s_ttft},
        "long": {"n": len(longs), "tpot_ms": l_tpot, "ttft_ms": l_ttft},
        "errors": len(errors),
        "results": {k: {kk: vv for kk, vv in v.items() if kk != "t_send"}
                    for k, v in results.items() if v},
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
