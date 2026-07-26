"""
anchor - a runtime grounding gate for LLM outputs.

Reads something an AI wrote, splits it into claims, and checks each claim
against your source documents. Grounded claims get a citation; made-up claims
get flagged. Spell-check, but for facts.

The core idea: one Verifier, interchangeable checkers. We ship two cheap ones,
each with different (and opposite) blind spots - which is exactly why the
pluggable interface matters:

  CoverageChecker  - pure Python, no deps. IDF-weighted share of the claim's
                     distinctive words found in the sources. Catches invented
                     or contradicted details; weak on heavy paraphrase.
  SemanticChecker  - sentence-transformers embeddings (optional dep). Catches
                     paraphrase; blind to numeric contradictions.

An AuditLog records every verdict as append-only JSONL - the evidence an
auditor points at (a system prompt is not evidence).
"""

from __future__ import annotations

import json
import math
import re
import time
import warnings
from dataclasses import dataclass
from typing import Protocol

warnings.filterwarnings("ignore")  # silence noisy transitive warnings


# --- model ------------------------------------------------------------------

@dataclass
class Source:
    text: str
    id: str


@dataclass
class Claim:
    text: str
    grounded: bool = False
    backed_by: str | None = None
    score: float = 0.0
    caught_by: str | None = None
    cite_doc: str | None = None
    cite_start: int = -1
    cite_end: int = -1


class Checker(Protocol):
    """A claim checker. Same contract, different strategies."""
    def check(self, claim: str, sources: list[Source]) -> Claim: ...


# --- text helpers -----------------------------------------------------------

_STOP = set("""
a an the to of in on at for with from by is are was were be been being and or
but if then than this that these those your you we they he she it his her their
our my me him them as so not no do does did has have had will would can could
should may might must into per up out over more most least also its set value
""".split())


_SENT_RE = re.compile(r"[^.!?\n]+[.!?]*")


def _spans(text: str) -> list[tuple[str, int, int]]:
    """Sentences with char offsets - the basis for exact citations."""
    return [(m.group(0).strip(), m.start(), m.end())
            for m in _SENT_RE.finditer(text) if m.group(0).strip()]


def _sentences(text: str) -> list[str]:
    return [s for s, _, _ in _spans(text)]


def chunk(text: str) -> list[dict]:
    """Split text into sentence chunks with char offsets - for evidence/citation UIs."""
    return [{"text": s, "start": a, "end": b} for s, a, b in _spans(text)]


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _stem(w: str) -> str:
    # ponytail: crude suffix stripper - days/day, requests/request, submitted/submit.
    for suf in ("ingly", "ing", "edly", "ed", "es", "ly", "er", "s"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def _content(text: str) -> list[str]:
    return [_stem(t) for t in _tokens(text) if t not in _STOP]


# --- negation / antonym helpers: a cheap "opposite direction" signal --------

_NEG_CUES = {"not", "no", "never", "none", "cannot", "cant", "neither", "nor",
             "without", "nobody", "nothing", "nowhere", "hardly", "barely"}
_NEG_PREFIXES = ("dis", "un", "im", "ir", "il", "non")


def _negated(text: str) -> bool:
    low = text.lower()
    toks = set(_tokens(text))
    return "n't" in low or any(c in toks for c in _NEG_CUES)


def _opposes(claim: str, source: str) -> bool:
    """Heuristic 'opposite direction' signal: a negation-cue mismatch between claim
    and source, or an antonym-prefix flip (dis-/un-/in-/... ). Catches 'do NOT need
    vs MUST' and 'honesty vs dishonesty' that lexical overlap alone can't see."""
    if _negated(claim) != _negated(source):
        return True
    ct, st = set(_tokens(claim)), set(_tokens(source))
    for w in ct:
        for p in _NEG_PREFIXES:
            if w.startswith(p) and len(w) > len(p) + 3 and w[len(p):] in st:
                return True
    for w in st:
        for p in _NEG_PREFIXES:
            if w.startswith(p) and len(w) > len(p) + 3 and w[len(p):] in ct:
                return True
    return False


# --- checker 1: coverage (pure Python, no deps) -----------------------------

class CoverageChecker:
    """IDF-weighted share of the claim's distinctive words found in sources."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def check(self, claim: str, sources: list[Source]) -> Claim:
        flat = [_content(sent) for s in sources for sent in _sentences(s.text)]
        if not flat:
            return Claim(claim, grounded=False, score=0.0)

        n = len(flat)
        df: dict[str, int] = {}
        for stems in flat:
            for t in set(stems):
                df[t] = df.get(t, 0) + 1
        idf = lambda w: math.log((1 + n) / (1 + df.get(w, 0))) + 1.0  # OOV -> rarest

        source_words = set().union(*[set(st) for st in flat])
        claim_words = _content(claim)
        if not claim_words:
            return Claim(claim, grounded=True, score=1.0)

        present = sum(idf(w) for w in claim_words if w in source_words)
        coverage = present / sum(idf(w) for w in claim_words)

        grounded = coverage >= self.threshold
        span = _best_span(claim, sources) if grounded else None
        if span:
            doc_id, sent, start, end = span
            return Claim(claim, grounded=True, score=coverage,
                         backed_by=f"[{doc_id}] {sent}",
                         cite_doc=doc_id, cite_start=start, cite_end=end)
        return Claim(claim, grounded=grounded, score=coverage)


# --- checker 2: semantic (optional sentence-transformers) -------------------

class SemanticChecker:
    """Embedding cosine. Catches paraphrase; blind to numeric contradictions."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", threshold: float = 0.55):
        self.model_name = model_name
        self.threshold = threshold
        self._model = None
        self._src_key = None
        self._src_labels: list[tuple[str, str]] = []
        self._src_embs = None

    def _ensure(self, sources: list[Source]) -> None:
        key = tuple((s.id, s.text) for s in sources)
        if self._src_key == key:
            return
        from sentence_transformers import SentenceTransformer  # lazy + optional
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        labels = [(s.id, sent) for s in sources for sent in _sentences(s.text)]
        self._src_labels = labels
        self._src_embs = self._model.encode(
            [sent for _, sent in labels], normalize_embeddings=True
        )
        self._src_key = key

    def check(self, claim: str, sources: list[Source]) -> Claim:
        self._ensure(sources)
        if not self._src_labels:
            return Claim(claim, grounded=False, score=0.0)
        cemb = self._model.encode([claim], normalize_embeddings=True)[0]
        sims = self._src_embs @ cemb
        i = int(sims.argmax())
        score = float(sims[i])
        sid, sent = self._src_labels[i]
        grounded = score >= self.threshold
        return Claim(claim, grounded=grounded,
                     backed_by=f"[{sid}] {sent}" if grounded else None, score=score)


# --- specialists + ensemble (patch a base checker's blind spots) ------------

_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def _numbers(text: str) -> list[str]:
    return _NUM_RE.findall(text)


class NumericSpecialist:
    """Catches numeric contradictions: a claim digit that appears nowhere in the
    sources. Fires only when the claim has digits; abstains otherwise. This is how
    values are really verified - extract and compare, not sentence similarity.

    Honest limit: compares against every number in the sources, so a value that
    coincidentally matches an unrelated topic can slip through; a stricter version
    scopes numbers to the claim's specific attribute (a v3 job)."""

    def contradicts(self, claim: str, sources: list[Source]) -> bool:
        claim_nums = _numbers(claim)
        if not claim_nums:
            return False  # abstain: no numeric assertion to check
        src_nums = set(n for s in sources for n in _numbers(s.text))
        return any(n not in src_nums for n in claim_nums)


class EnsembleChecker:
    """A base checker, overridden by specialists that patch its blind spots.
    Grounded only if the base says grounded AND no specialist flags a contradiction."""

    def __init__(self, base: Checker, specialists: list | None = None):
        self.base = base
        self.specialists = specialists if specialists is not None else [NumericSpecialist()]

    def check(self, claim: str, sources: list[Source]) -> Claim:
        c = self.base.check(claim, sources)
        for sp in self.specialists:
            if sp.contradicts(claim, sources):
                return Claim(claim, grounded=False, backed_by=None, score=c.score)
        return c


# --- v3: NLI checker, LLM judge, and the cost-saving cascade ----------------

def _best_span(claim: str, sources: list[Source]):
    """The source sentence (doc id + char offsets) most overlapping the claim."""
    claim_set, best, best_ov = set(_content(claim)), None, -1
    for s in sources:
        for sent, start, end in _spans(s.text):
            ov = len(claim_set & set(_content(sent)))
            if ov > best_ov:
                best_ov, best = ov, (s.id, sent, start, end)
    return best


class _SemIndex:
    """Embeds source sentences once; retrieves the best by cosine to the claim.
    More robust than lexical overlap when the corpus has word-similar decoy docs."""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None
        self._key = None
        self._labels = []
        self._embs = None

    def best_span(self, claim: str, sources: list[Source]):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        key = tuple((s.id, s.text) for s in sources)
        if self._key != key:
            self._labels = [(s.id, sent, st, en)
                            for s in sources for sent, st, en in _spans(s.text)]
            self._embs = self._model.encode([sent for _, sent, _, _ in self._labels],
                                            normalize_embeddings=True)
            self._key = key
        if not self._labels:
            return None
        cemb = self._model.encode([claim], normalize_embeddings=True)[0]
        i = int((self._embs @ cemb).argmax())
        return self._labels[i]


class NLIChecker:
    """Local transformer (NLI): labels a claim vs its best-matching source sentence
    as entailment => grounded, contradiction/neutral => not. Catches binding and
    contradiction cases that lexical/embedding checkers are blind to. Runs locally
    - no API, private. Picks the best model the environment supports: DeBERTa-v3
    FEVER-NLI if sentencepiece is installed, else the lighter distilbert-mnli.
    `pip install sentencepiece` upgrades to the stronger model."""

    def __init__(self, model_name: str | None = None, lex_fallback: bool = False,
                 negation_gate: bool = False, retriever: str = "lexical"):
        self.model_name = model_name
        self.lex_fallback = lex_fallback
        self.negation_gate = negation_gate
        self.retriever = retriever
        self._pipe = None
        self._unavailable = False
        self._lex = None
        self._sem = _SemIndex() if retriever == "semantic" else None

    def _lex_grounded(self, claim: str, sources: list[Source]) -> bool:
        """Lexical overlap check, used only to clear NLI 'neutral' paraphrase false-alarms.
        Strict threshold: only clear when the claim is STRONGLY covered (a likely paraphrase),
        so a partial-overlap fabrication (e.g. unsupported 'extra 3 PTO days for tenure') stays flagged."""
        if self._lex is None:
            self._lex = CoverageChecker(threshold=0.65)
        return self._lex.check(claim, sources).grounded

    def available(self) -> bool:
        return not self._unavailable

    def _ensure(self) -> None:
        if self._pipe is not None or self._unavailable:
            return
        import importlib
        from transformers import pipeline
        candidates = []
        if self.model_name:
            candidates.append(self.model_name)
        elif importlib.util.find_spec("sentencepiece"):
            candidates.append("MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
        candidates.append("typeform/distilbert-base-uncased-mnli")
        for name in candidates:
            try:
                self._pipe = pipeline("text-classification", model=name, top_k=3)
                self.model_name = name
                return
            except Exception:
                continue
        self._unavailable = True

    def check(self, claim: str, sources: list[Source]) -> Claim:
        self._ensure()
        if self._unavailable:
            return Claim(claim, grounded=True, score=1.0)  # abstain -> let other tiers decide
        span = self._sem.best_span(claim, sources) if self._sem else _best_span(claim, sources)
        if not span:
            return Claim(claim, grounded=False, score=0.0)
        doc_id, premise, start, end = span
        out = self._pipe({"text": premise, "text_pair": claim})
        probs = {d["label"].lower(): d["score"] for d in out}
        ent = probs.get("entailment", 0.0)
        contra = probs.get("contradiction", 0.0)
        neutral = probs.get("neutral", 0.0)
        if ent >= contra and ent >= neutral:
            grounded = True                            # entailment: reliable
        elif contra >= neutral:
            grounded = False                           # contradiction: reliable
        elif self.lex_fallback:
            # neutral: only clear if NO opposition signal AND good lexical overlap
            opposes = contra >= 0.30 or (self.negation_gate and _opposes(claim, premise))
            grounded = (not opposes) and self._lex_grounded(claim, sources)
        else:
            grounded = False                           # neutral, no fallback -> unsupported
        if grounded:
            return Claim(claim, grounded=True, score=ent, backed_by=f"[{doc_id}] {premise}",
                         cite_doc=doc_id, cite_start=start, cite_end=end)
        return Claim(claim, grounded=False, score=ent)


class LLMJudgeChecker:
    """Top tier: a frontier LLM judges claim vs sources. Strongest checker; costs
    tokens + latency and (hosted API) sends data off-prem - so the cascade only
    calls it on the hard residue. Skips gracefully when no key is configured."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self._client = None

    def available(self) -> bool:
        import os
        return bool(os.getenv("OPENAI_API_KEY"))

    def check(self, claim: str, sources: list[Source]) -> Claim:
        from openai import OpenAI
        if self._client is None:
            self._client = OpenAI()
        src = "\n".join(f"[{s.id}] {s.text}" for s in sources)
        prompt = (f"Sources:\n{src}\n\nClaim: {claim}\n\n"
                  f"Is the claim fully supported by the sources? "
                  f"Reply with one word: SUPPORTED or UNSUPPORTED.")
        resp = self._client.chat.completions.create(
            model=self.model, temperature=0,
            messages=[{"role": "user", "content": prompt}])
        verdict = resp.choices[0].message.content.strip().upper()
        grounded = verdict.startswith("SUPPORTED")
        return Claim(claim, grounded=grounded, score=1.0 if grounded else 0.0,
                     backed_by="[llm-judge]" if grounded else None)


class CascadeChecker:
    """Run checkers cheapest-first; stop and flag the moment one says ungrounded.
    Expensive tiers only run on the residue the cheap ones couldn't fault.
    Records which tier caught a flagged claim in Claim.caught_by."""

    def __init__(self, layers: list, name: str = "Cascade"):
        self.layers = layers
        self.name = name

    def check(self, claim: str, sources: list[Source]) -> Claim:
        for layer in self.layers:
            if not getattr(layer, "available", lambda: True)():
                continue  # skip tiers not configured (NLI dep missing, no API key)
            c = layer.check(claim, sources)
            if not c.grounded:
                return Claim(claim, grounded=False, backed_by=None, score=c.score,
                             caught_by=type(layer).__name__)
        return Claim(claim, grounded=True, score=1.0)


class ConfirmedChecker:
    """The recommended core: a primary verifier (NLI) for recall, plus a confirmer
    (LLM judge) that vets each FLAGGED claim to clear false alarms -> precision.
    The judge runs only on claims the primary flagged, so cost is bounded to the
    flagged set. Prefer this over the OR-cascade (CascadeChecker), which accumulates
    false alarms instead of clearing them."""

    def __init__(self, primary: Checker, confirmer: Checker):
        self.primary = primary
        self.confirmer = confirmer

    def available(self) -> bool:
        return (getattr(self.primary, "available", lambda: True)()
                and getattr(self.confirmer, "available", lambda: True)())

    def check(self, claim: str, sources: list[Source]) -> Claim:
        c = self.primary.check(claim, sources)
        if c.grounded:
            return c                                   # primary passed -> trust it
        if not getattr(self.confirmer, "available", lambda: True)():
            return c                                   # primary flagged but no confirmer -> keep the flag
        return self.confirmer.check(claim, sources)    # primary flagged -> judge vets


# --- audit ------------------------------------------------------------------

class AuditLog:
    """Append-only JSONL audit trail - one record per verdict."""

    def __init__(self, path: str | None = None, mode: str = "a"):
        self.path = path
        self._f = open(path, mode) if path else None
        self.records: list[dict] = []

    def log(self, claim: str, grounded: bool, score: float, checker: str,
            backed_by: str | None = None) -> dict:
        rec = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "checker": checker,
            "claim": claim,
            "verdict": "grounded" if grounded else "UNGROUNDED",
            "score": round(float(score), 4),
            "backed_by": backed_by,
        }
        self.records.append(rec)
        if self._f:
            self._f.write(json.dumps(rec) + "\n")
            self._f.flush()
        return rec

    def close(self) -> None:
        if self._f:
            self._f.close()


# --- verifier ---------------------------------------------------------------

def _decompose(sentence: str) -> list[str]:
    """Split a compound sentence into sub-claims on conjunctions (' and ', '; ').
    Conservative: only splits when both halves are >=3 words (avoids 'R&D',
    'salt and pepper'). Recursive: 'A and B and C' -> ['A', 'B', 'C']."""
    parts = re.split(r"\s+(?:and|;)\s+", sentence, maxsplit=1)
    if len(parts) == 2 and len(parts[0].split()) >= 3 and len(parts[1].split()) >= 3:
        return _decompose(parts[0].strip()) + _decompose(parts[1].strip())
    return [sentence]


class Verifier:
    def __init__(self, sources: list[Source], checker: Checker | None = None,
                 audit: AuditLog | None = None, checker_name: str = ""):
        self.sources = sources
        self.checker = checker or CoverageChecker()
        self.audit = audit
        self.checker_name = checker_name or type(self.checker).__name__

    def verify(self, output: str) -> list[Claim]:
        claims = []
        for sentence in _sentences(output):
            for sub in _decompose(sentence):
                claims.append(self.checker.check(sub, self.sources))
        if self.audit:
            for c in claims:
                self.audit.log(c.text, c.grounded, c.score, self.checker_name, c.backed_by)
        return claims


# --- citation layer: turn an answer into a cited, flagged answer ------------

@dataclass
class Citation:
    n: int
    doc_id: str
    start: int
    end: int
    snippet: str


@dataclass
class AnnotatedAnswer:
    text: str           # the output, with inline [n] citations and ⚠️ markers
    citations: list     # list[Citation] - exact doc + char span per grounded claim
    flags: list         # list[str] - ungrounded claim texts


def _snippet(c: Claim, sources: list[Source]) -> str | None:
    src = next((s for s in sources if s.id == c.cite_doc), None)
    return src.text[c.cite_start:c.cite_end] if src and c.cite_start >= 0 else None


def annotate(output: str, sources: list[Source], checker: Checker | None = None) -> AnnotatedAnswer:
    """Verify each claim and return the output with inline [n] citations pointing
    to the exact source doc + char span, plus ⚠️ markers on ungrounded claims.
    Default checker is the NLI core; pass any Checker to use another."""
    claims = Verifier(sources, checker or NLIChecker()).verify(output)
    citations, parts, flags = [], [], []
    for c in claims:
        if c.grounded and c.cite_doc:
            n = len(citations) + 1
            snip = (_snippet(c, sources) or "").strip()
            citations.append(Citation(n, c.cite_doc, c.cite_start, c.cite_end, snip))
            parts.append(f"{c.text} [{n}]")
        elif c.grounded:
            parts.append(c.text)
        else:
            parts.append(f"{c.text} ⚠️")
            flags.append(c.text)
    return AnnotatedAnswer(" ".join(parts), citations, flags)


def render_markdown(ans: AnnotatedAnswer) -> str:
    lines = [ans.text, "", "Citations:"]
    for c in ans.citations:
        lines.append(f"  [{c.n}] {c.doc_id}:{c.start}-{c.end}  \"{c.snippet}\"")
    if ans.flags:
        lines.append("\nFlagged (not found in any source):")
        for f in ans.flags:
            lines.append(f"  ⚠️ {f}")
    return "\n".join(lines)
