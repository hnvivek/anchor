"""examples/usage.py - how to use anchor in your own code.

Install once (from the repo root):  pip install -e .
Then:  python examples/usage.py
"""
from anchor import NLIChecker, Source, Verifier, annotate, render_markdown

# 1. The source document(s) the AI should have drawn from (the ground truth).
sources = [
    Source(id="pto", text=(
        "Employees get 15 days of PTO per year. "
        "Requests must be submitted 7 days in advance.")),
]

# 2. The AI-generated answer you want to fact-check.
answer = (
    "Employees get 15 days of PTO per year. "
    "Staff also receive a free puppy on their birthday.")

# 3. Verify every claim. Grounded claims get a citation to the exact source span;
#    unsupported claims are flagged.
result = annotate(answer, sources)          # uses the NLI core by default
print(render_markdown(result))
# result.text       -> the answer with inline [n] citations and warning flags
# result.citations  -> [{n, doc_id, start, end, snippet}, ...]
# result.flags      -> [claims not found in any source]

# Or check a single claim directly:
claim = Verifier(sources, NLIChecker()).verify("Staff get 25 days of PTO.")[0]
print(f"\n'{claim.text}' -> {'grounded' if claim.grounded else 'UNGROUNDED'} "
      f"(score {claim.score:.2f})")
