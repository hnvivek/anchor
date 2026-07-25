"""bench_fever - run anchor on the FEVER public fact-checking benchmark.

FEVER (Fact Extraction and VERification): each item is a claim + gold evidence
sentences, labelled SUPPORTS / REFUTES / NOT ENOUGH INFO. We score anchor on the
*verifiable* subset: SUPPORTS (-> grounded) vs REFUTES (-> not). A real, public,
independent test - not our hand-written eval.

    run:  python3 bench_fever.py
"""
from __future__ import annotations

import random
import statistics
import time

from datasets import load_dataset

from anchor import NLIChecker, Source, Verifier

N = 200  # balanced sample (N/2 SUPPORTS, N/2 REFUTES)


def evidence_text(ev) -> str:
    """Flatten FEVER's nested evidence into one source string."""
    sents = []
    for item in (ev or []):
        if item is None:
            continue
        if item and isinstance(item[0], (list, tuple)):
            for s in item:
                if len(s) >= 3:
                    sents.append(str(s[2]))
        elif len(item) >= 3:
            sents.append(str(item[2]))
    return " ".join(sents)


def main():
    ds = load_dataset("copenlu/fever_gold_evidence", split="train")
    sup, ref = [], []
    for r in ds:
        if r.get("verifiable") != "VERIFIABLE" or r["label"] not in ("SUPPORTS", "REFUTES"):
            continue
        ev = evidence_text(r["evidence"])
        if not ev:
            continue
        (sup if r["label"] == "SUPPORTS" else ref).append((r["claim"], ev))

    random.seed(7)
    random.shuffle(sup)
    random.shuffle(ref)
    k = N // 2
    sample = [(c, e, True) for c, e in sup[:k]] + [(c, e, False) for c, e in ref[:k]]
    random.shuffle(sample)

    chk = NLIChecker(lex_fallback=True, negation_gate=True)
    tp = tn = fp = fn = 0
    lat = []
    for claim, ev, expected in sample:
        t0 = time.perf_counter()
        pred = Verifier([Source(id="evidence", text=ev)], chk).verify(claim)[0].grounded
        lat.append((time.perf_counter() - t0) * 1000)
        if expected and pred:          # SUPPORTS, anchor grounded -> correct
            tn += 1
        elif expected:                 # SUPPORTS, anchor flagged -> false alarm
            fp += 1
        elif not pred:                 # REFUTES, anchor flagged -> caught
            tp += 1
        else:                          # REFUTES, anchor grounded -> missed
            fn += 1

    n = len(sample)
    acc = (tp + tn) / n
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    print(f"FEVER (copenlu/fever_gold_evidence, train) - verifiable SUPPORTS vs REFUTES, n={n}")
    print(f"  accuracy {acc:.2f}  precision {prec:.2f}  recall {rec:.2f}  f1 {f1:.2f}"
          f"  (median {statistics.median(lat):.0f} ms/claim)")
    print(f"  TP={tp} refutes caught · TN={tn} supports passed · "
          f"FP={fp} supports wrongly flagged · FN={fn} refutes missed")


if __name__ == "__main__":
    main()
