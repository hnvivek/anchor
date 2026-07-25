# ⚓ anchor

**A runtime grounding gate for LLM outputs — catch hallucinated claims before they ship, with exact citations.**

Spell-check, but for facts. Give `anchor` an AI-generated answer and the source documents it should have drawn from: it splits the answer into atomic claims, verifies each against the sources, attaches an **inline citation** to every grounded claim (pointing to the exact `doc_id:start-end`), and flags the ones that aren't supported. **0.93 F1 on the FEVER public benchmark, fully local, no API.**

## Benchmark

**FEVER** (the standard public fact-checking benchmark — claim + gold evidence → SUPPORTS/REFUTES, n=200):

| | accuracy | precision | recall | F1 |
|---|---|---|---|---|
| **anchor (NLI + lex + neg)** | **0.93** | 0.90 | **0.95** | **0.93** |

Plus our own 27-case stress set (paraphrase, fabrication, numeric, negation, entity-swap, conditionals, multi-doc, dates/units):

| checker | acc | prec | recall | F1 |
|---|---|---|---|---|
| NLI-only | 0.85 | 0.82 | 1.00 | 0.90 |
| **NLI + lex + neg** | **0.96** | **0.95** | **1.00** | **0.97** |

## Why

Telling an LLM *"only answer from the sources"* helps — but it doesn't **guarantee** or **prove** anything. The model still invents, and you can't show an auditor "this was grounded." `anchor` is the independent check **plus** the citation/audit trail.

- **vs RAGAS** — RAGAS grades your RAG pipeline *offline* over a test set. `anchor` checks a *live* output at runtime, per claim, with citations.
- **vs NeMo Guardrails** — NeMo is a broad behavioral-rails framework (Colang, an LLM call per rail). `anchor` is a focused, **cheap-first** grounding layer: local models, an audit trail, and the LLM judge only on the residue.

## Quick start

```bash
git clone https://github.com/hnvivek/anchor && cd anchor
pip install -r requirements.txt          # NLI core (sentence-transformers, etc.)

python cite.py            # verify an AI answer against a source doc
python eval.py            # the stress-test eval (precision / recall / latency)
python bench_fever.py     # the FEVER public benchmark
python bench.py           # benchmark on bench.json — or YOUR own dataset
```

### Playground UI

```bash
pip install -r playground/requirements.txt
cd playground && uvicorn app:app --reload      # http://localhost:8000
```

The UI highlights the answer **inline** — grounded claims underlined **blue** (hover → the cited source quote), unsupported claims underlined **red**. Deploy to any container PaaS — see `Dockerfile` / `render.yaml`. Set `ANCHOR_CHECKER=coverage` for a zero-model, low-RAM mode on small hosting tiers.

## How it works

Each claim is verified through a pluggable checker interface:

| checker | what it catches | cost |
|---|---|---|
| **CoverageChecker** | invented details (lexical/IDF overlap) | pure Python, free |
| **NLIChecker** (DeBERTa) | paraphrase, binding/entity-swap, numeric, negation, conditionals | local, ~15 ms |
| **NumericSpecialist / negation gate** | wrong numbers; "opposite direction" (not/must, dis-/un-) | pure Python, free |
| **LLMJudgeChecker** | clears the last false alarms (precision confirmer) | optional, your key |

The recommended core is **NLI + lex + neg**: NLI's entailment/contradiction are reliable; when NLI is unsure ("neutral"), a **lexical confirmer** clears paraphrase false-alarms and a **negation/antonym gate** keeps "do **not** need vs **must**" flagged. Every grounded claim returns an exact citation; every verdict appends to an append-only JSONL audit log.

## Architecture

```
anchor.py            # the library — importable, minimal deps. The product.
eval.py / bench.py / bench_fever.py / cite.py   # tooling & benchmarks
bench.json           # sample benchmark (swappable for your own data)
playground/          # FastAPI app + UI — separate deps, thin layer over the library
Dockerfile / render.yaml   # one-command container deploy
```

`anchor` is the **verification** layer, not a retriever — point it at your existing corpus (RAG retriever, Confluence, SharePoint, vector DB) and it verifies + cites.

## Roadmap

- [x] Citation layer — inline citations with exact `doc:start-end`
- [x] Local NLI core (DeBERTa) + lexical & negation confirmers
- [x] Public benchmark on FEVER (0.93 F1)
- [ ] Provider-agnostic + local judge (Ollama) for air-gapped deployments
- [ ] Human-in-the-loop gate (block / strip / flag-for-review)

## License

MIT
