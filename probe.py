"""probe - run ONE (source, claim) case through every checker.
Edit SOURCE/CLAIM below, then:  python3 probe.py
"""
from anchor import Source, Verifier, CoverageChecker, SemanticChecker, EnsembleChecker

SOURCE = Source(id="input", text="Hi, I am 21 years old and my wife is 18 years old.")
CLAIM = "Hi, I am 18 years old."

print("SOURCE:", SOURCE.text)
print("CLAIM :", CLAIM, "\n")
for name, chk in [("Coverage", CoverageChecker()),
                  ("Semantic", SemanticChecker()),
                  ("Coverage+Numeric", EnsembleChecker(CoverageChecker()))]:
    c = Verifier([SOURCE], chk).verify(CLAIM)[0]
    mark = "✅ grounded" if c.grounded else "❌ UNGROUNDED"
    print(f"  {name:<18} {mark}  ({c.score:.2f})")
