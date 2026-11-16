"""Measure the voice custom-LLM endpoint the way a voice agent experiences it.

For each caller utterance it streams POST {base}/chat/completions and records
  first_token: request sent -> first spoken text chunk (the number that decides whether
               the call feels natural; the voice platform adds speech-to-text and
               text-to-speech time on top)
  total:       request sent -> end of stream
then prints p50 / p95 over all runs.

Usage:
  VOICE_SPIKE_SECRET=... venv/bin/python scripts/voice_latency_probe.py \
      --base https://api.aichatsupport.my/api/v1/voice/spike/12 --runs 3
Use a bot that has a real knowledge base; run once cold and once warm, and compare models by
changing the bot's chat model / package fallback chain between runs.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import httpx

UTTERANCES = [
    "Hi, what are your opening hours?",
    "How much does delivery cost to Perth?",
    "I want to return something I bought last week, how does that work?",
    "Can I speak to someone about my order?",
    "Do you offer a warranty on your products?",
    "Where are you located?",
    "What payment methods do you accept?",
    "Sorry, can you say that again?",
]


def probe(client: httpx.Client, url: str, secret: str, text: str) -> tuple[float | None, float, str]:
    body = {"stream": True, "messages": [{"role": "user", "content": text}]}
    t0 = time.perf_counter()
    first: float | None = None
    spoken: list[str] = []
    with client.stream("POST", url, json=body, headers={"Authorization": f"Bearer {secret}"}) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            piece = json.loads(line[6:])["choices"][0]["delta"].get("content") or ""
            if piece:
                if first is None:
                    first = time.perf_counter() - t0
                spoken.append(piece)
    return first, time.perf_counter() - t0, "".join(spoken)


def pct(values: list[float], p: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, round(p / 100 * (len(s) - 1)))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="e.g. https://api.aichatsupport.my/api/v1/voice/spike/12")
    ap.add_argument("--runs", type=int, default=3, help="passes over the utterance list")
    ap.add_argument("--secret", default=os.getenv("VOICE_SPIKE_SECRET", ""))
    args = ap.parse_args()
    if not args.secret:
        print("Set VOICE_SPIKE_SECRET or pass --secret", file=sys.stderr)
        return 2
    url = args.base.rstrip("/") + "/chat/completions"
    firsts: list[float] = []
    totals: list[float] = []
    with httpx.Client(timeout=60) as client:
        for n in range(args.runs):
            for text in UTTERANCES:
                first, total, spoken = probe(client, url, args.secret, text)
                if first is not None:
                    firsts.append(first)
                totals.append(total)
                print(f"[{n + 1}] first={'%.0f' % (first * 1000) if first is not None else '  -'}ms "
                      f"total={total * 1000:.0f}ms  {text[:38]!r} -> {spoken[:60]!r}")
    if not firsts:
        print("No spoken text received", file=sys.stderr)
        return 1
    print(f"\nfirst token  p50={pct(firsts, 50) * 1000:.0f}ms  p95={pct(firsts, 95) * 1000:.0f}ms  "
          f"mean={statistics.mean(firsts) * 1000:.0f}ms  (n={len(firsts)})")
    print(f"full reply   p50={pct(totals, 50) * 1000:.0f}ms  p95={pct(totals, 95) * 1000:.0f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
