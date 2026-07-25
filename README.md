# ⚓ anchor

**A runtime grounding gate for LLM outputs — catch hallucinated claims before they ship, with exact citations.**

Spell-check, but for facts. Give `anchor` an AI-generated answer and the source documents it should have drawn from: it splits the answer into atomic claims, verifies each against the sources, attaches an **inline citation** to every grounded claim (pointing to the exact `doc_id:start-end`), and flags the ones that aren't supported.

```
"You can roll over up to 5 unused days into next year. [1]
 Employees with 5+ years of tenure qualify for 3 bonus vacation days. ⚠️"

[1] PTO-Policy:38-88  "Up to 5 unused days roll over into the next year."
⚠️ not found in any source
```

## Why

Telling an LLM *"only answer from the sources"* helps — but it doesn't **guarantee** or **prove** anything. The model still invents, and you can't show an auditor "this was grounded." `anchor` is the independent check **plus** the citation/audit trail.

- **vs RAGAS** — RAGAS grades your RAG pipeline *offline* over a test set. `anchor` checks a *live* output at runtime, per claim, with citations.
- **vs NeMo Guardrails** — NeMo is a broad behavioral-rails framework (Colang, an LLM call per rail). `anchor` is a focused, **cheap-first** grounding layer: local models, an audit trail, and the LLM judge only on the residue.

## Quick start

```bash
git clone https://github.com/hnvivek/anchor && cd anchor
pip install -r requirements.txt          # NLI core (sentence-transformers, etc.)

python cite.py        # verify an AI answer against a source doc
python eval.py        # the stress-test eval (precision / recall / latency)
python bench.py       # benchmark on bench.json — or YOUR own dataset
```

### Playground UI

```bash
pip install -r playground/requirements.txt
cd playground && uvicorn app:app --reload      # http://localhost:8000
```

Deploy to any container PaaS — see `Dockerfile` / `render.yaml`. Set `ANCHOR_CHECKER=coverage` for a zero-model, low-RAM mode on small hosting tiers.

## How it works

Each claim is verified through a pluggable checker interface:

| checker | what it catches | cost |
|---|---|---|
| **CoverageChecker** | invented details (lexical/IDF overlap) | pure Python, free |
| **NLIChecker** (DeBERTa) | paraphrase, binding/entity-swap, numeric, negation, conditionals | local, ~30 ms |
| **LLMJudgeChecker** | clears NLI's false alarms (precision confirmer) | optional, your key |

The eval proved an OR-cascade just **accumulates** false alarms, so the recommended core is **NLI-only** (recall 1.00) with the **judge as a confirmer** — not a pile of OR'd layers. Every grounded claim returns an exact citation; every verdict appends to an append-only JSONL audit log.

## Eval (honest)

27-case stress set — paraphrase, fabrication, numeric, entity-swap, negation, conditionals, multi-doc conflation, dates/units:

| checker | acc | prec | recall | f1 |
|---|---|---|---|---|
| Cov+Num | 0.52 | 0.73 | 0.44 | 0.55 |
| **NLI-only** | **0.85** | 0.82 | **1.00** | **0.90** |

NLI catches **every lie locally**; its only weakness is *precision* on loose paraphrases — which the judge confirmer clears. `python eval.py` reproduces this; `bench.py` scores `anchor` on **any** dataset (point it at your own JSON, or FEVER/TruthfulQA in the same format).

## Architecture

```
anchor.py            # the library — importable, minimal deps. The product.
eval.py / bench.py / cite.py / probe.py   # tooling & examples
bench.json           # sample benchmark (swappable for your own data)
playground/          # FastAPI app + UI — separate deps, thin layer over the library
Dockerfile / render.yaml   # one-command container deploy
```

## Roadmap

- [x] Citation layer — inline `[n]` citations with exact `doc:start-end`
- [x] Local NLI core (DeBERTa) + judge confirmer
- [ ] Public benchmark on FEVER / TruthfulQA
- [ ] Provider-agnostic + local judge (Ollama) for air-gapped deployments
- [ ] Human-in-the-loop gate (block / strip / flag-for-review)

## License

MIT
