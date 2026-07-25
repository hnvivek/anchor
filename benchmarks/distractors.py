"""distractors - does lexical retrieval get fooled by decoy docs? Does semantic fix it?

Each case is a claim + the CORRECT source + DECOY sources that share words but mean
something different. A robust retriever must still pick the correct source. We score
lexical (word overlap) vs semantic (embeddings) retrieval.

    run:  python3 benchmarks/distractors.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anchor import NLIChecker, Source, Verifier

# (claim, correct source, decoy sources, expected_grounded)
CASES = [
    ("The retry_timeout default is 3000 seconds.",
     Source(id="retry", text="The default value of retry_timeout is 3000, measured in seconds. The max is 10000 seconds."),
     [Source(id="session", text="The default session timeout is 3000 milliseconds. Sessions expire after this period.")],
     True),
    ("Employees can roll over up to 5 unused PTO days.",
     Source(id="pto", text="Up to 5 unused PTO days roll over into the next year. Days beyond 5 are forfeited."),
     [Source(id="remote", text="Unused remote-work days do not roll over. Remote days reset each quarter.")],
     True),
    ("Refunds are available within 30 days of purchase.",
     Source(id="refund", text="Refunds are available within 30 days of purchase. Issued to the original payment method."),
     [Source(id="trial", text="The free trial lasts 30 days. Trials cancel automatically after 30 days.")],
     True),
    ("The Pro plan costs 49 dollars per user per month.",
     Source(id="pricing", text="The Pro plan costs 49 dollars per user per month. A 14-day trial is available."),
     [Source(id="expense", text="Lodging reimbursement is capped at 49 dollars per night for extended stays.")],
     True),
    ("Customer data is retained for 7 years.",
     Source(id="retention", text="Customer data is retained for 7 years after account closure, per regulation."),
     [Source(id="deploy", text="Deployment logs are retained for 7 years. Logs older than 7 years are archived.")],
     True),
]


def score(retriever: str):
    chk = NLIChecker(lex_fallback=True, negation_gate=True, retriever=retriever)
    correct = 0
    for claim, correct_doc, decoys, expected in CASES:
        srcs = [correct_doc] + decoys
        c = Verifier(srcs, chk).verify(claim)[0]
        ok = (c.grounded == expected) and (c.cite_doc == correct_doc.id)
        correct += ok
        print(f"  [{retriever:<9}] cited={str(c.cite_doc):<10} grounded={str(c.grounded):<5} "
              f"{'OK ' if ok else 'MISS'}  {claim[:46]}")
    return correct


def main():
    print("Lexical retrieval (word overlap), decoys present:")
    lc = score("lexical")
    print("\nSemantic retrieval (embeddings), decoys present:")
    sc = score("semantic")
    n = len(CASES)
    print(f"\nWith distractor docs:  lexical {lc}/{n}   |   semantic {sc}/{n}")
    print("(semantic retrieval should cite the correct doc even when a decoy shares words)")


if __name__ == "__main__":
    main()
