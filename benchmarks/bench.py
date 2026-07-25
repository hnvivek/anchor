#!/usr/bin/env python3
"""bench - score anchor on any dataset. The 'prove it on YOUR data' harness.

    python3 bench.py              # the shipped sample benchmark (bench.json)
    python3 bench.py mycases.json # YOUR cases

Case format (a JSON list):
    {"category": "...", "sources": [{"id":"...","text":"..."}],
     "claim": "...", "expected": true | false}

To benchmark on a PUBLIC dataset (FEVER, TruthfulQA, a RAG-faithfulness set),
export its (claim, evidence, label) rows into this JSON format, then run bench.
Reports accuracy / precision / recall / F1, per-category accuracy, and latency.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

from anchor import (CoverageChecker, EnsembleChecker, NLIChecker, Source, Verifier)


def load_cases(path: str):
    with open(path) as f:
        rows = json.load(f)
    out = []
    for r in rows:
        srcs = [Source(id=s["id"], text=s["text"]) for s in r["sources"]]
        out.append((r.get("category", "?"), srcs, r["claim"], bool(r["expected"])))
    return out


def score(cases, make, label):
    chk = make()
    tp = tn = fp = fn = 0
    by: dict = {}
    lat = []
    for cat, sources, claim, exp in cases:
        t0 = time.perf_counter()
        pred = Verifier(sources, chk).verify(claim)[0].grounded
        lat.append((time.perf_counter() - t0) * 1000)
        d = by.setdefault(cat, {"ok": 0, "n": 0})
        d["n"] += 1
        if pred == exp:
            d["ok"] += 1
        if exp is False and pred is False:
            tp += 1          # caught a real lie
        elif exp is False:
            fn += 1          # missed a lie
        elif pred is False:
            fp += 1          # false alarm
        else:
            tn += 1          # correctly passed
    n = len(cases)
    acc = (tp + tn) / n
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"label": label, "acc": acc, "prec": prec, "rec": rec, "f1": f1,
            "med": statistics.median(lat) or 0.0, "by": by}


def main():
    default = str(Path(__file__).resolve().parent / "bench.json")
    path = sys.argv[1] if len(sys.argv) > 1 else default
    cases = load_cases(path)
    print(f"anchor benchmark  -  {path}  ({len(cases)} cases)\n")
    results = [score(cases, lambda: EnsembleChecker(CoverageChecker()), "Cov+Num"),
               score(cases, lambda: NLIChecker(), "NLI-only")]
    print(f"  {'checker':<10}{'acc':>6}{'prec':>7}{'rec':>7}{'f1':>6}{'ms':>8}")
    print("  " + "-" * 42)
    for r in results:
        print(f"  {r['label']:<10}{r['acc']:>6.2f}{r['prec']:>7.2f}{r['rec']:>7.2f}"
              f"{r['f1']:>6.2f}{r['med']:>8.1f}")
    print("\n  per-category (NLI-only):")
    for cat in sorted({c for c, *_ in cases}):
        b = results[1]["by"][cat]
        print(f"  {cat:<26}{b['ok']/b['n']:>5.2f}  ({b['ok']}/{b['n']})")
    print("\n  Point bench.py at YOUR cases (or FEVER/TruthfulQA in this JSON format).")


if __name__ == "__main__":
    main()
