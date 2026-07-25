"""anchor playground - a FastAPI app serving the citation-layer UI.

A thin layer over the anchor library. Lives in playground/ so the published
library stays free of server/UI dependencies.

    run locally:  cd playground && uvicorn app:app --reload   (http://localhost:8000)
    deploy:       container PaaS (see ../Dockerfile, ../render.yaml)

Set ANCHOR_CHECKER=coverage for a zero-model, low-RAM mode (small hosting tiers);
default is the local NLI core.
"""
from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# make the sibling `anchor` library importable when run from playground/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from anchor import CoverageChecker, NLIChecker, Source, Verifier, chunk

_HERE = Path(__file__).parent

SAMPLES = [
    {"id": "pto", "name": "PTO Policy",
     "text": "Employees accrue 15 days of paid time off per year. Up to 5 unused days "
             "roll over into the next year. PTO requests must be submitted at least 7 days in advance."},
    {"id": "remote", "name": "Remote Work Policy",
     "text": "Employees may work remotely up to three days per week. Core hours are 10am to 3pm "
             "Eastern. A stable internet connection of at least 25 Mbps is required for remote work."},
    {"id": "expense", "name": "Expense Reimbursement",
     "text": "Travel expenses are reimbursed up to 200 dollars per day for lodging. Meal expenses "
             "are capped at 75 dollars per day. All expenses require a receipt and manager approval."},
    {"id": "rate", "name": "API Rate Limits",
     "text": "The public API allows 1000 requests per minute per API key. Exceeding the limit returns "
             "a 429 status code. Clients should implement exponential backoff when retrying."},
    {"id": "auth", "name": "Authentication",
     "text": "All API requests require a bearer token in the Authorization header. Tokens expire after "
             "24 hours. Use the refresh token endpoint to obtain a new access token."},
    {"id": "retry", "name": "SDK retry_timeout",
     "text": "The retry_timeout parameter sets how long the client waits before retrying a failed request. "
             "The default value is 3000, measured in seconds."},
    {"id": "deploy", "name": "Deployment Runbook",
     "text": "Production deploys happen on Tuesdays and Thursdays. A deploy requires approval from two "
             "reviewers. Rollback must be initiated within 15 minutes of a failed deploy."},
    {"id": "oncall", "name": "On-call Rotation",
     "text": "The on-call rotation cycles weekly across the engineering team. The primary on-call responds "
             "to pages within 5 minutes. The secondary on-call is the backup."},
    {"id": "pricing", "name": "Product Pricing",
     "text": "The Pro plan costs 49 dollars per user per month. The Enterprise plan requires an annual "
             "contract. A 14-day free trial is available for new accounts."},
    {"id": "refund", "name": "Refund Policy",
     "text": "Refunds are available within 30 days of purchase. Refunds are issued to the original payment "
             "method. Subscriptions canceled mid-cycle are not refunded."},
    {"id": "retention", "name": "Data Retention",
     "text": "Customer data is retained for 7 years after account closure. Backup data is retained for "
             "90 days. Data deletion requests are processed within 30 days."},
    {"id": "incident", "name": "Incident Response",
     "text": "Severity 1 incidents require a response within 15 minutes. The incident commander coordinates "
             "the response. A postmortem is required for all severity 1 and 2 incidents."},
]
SAMPLE_ANSWER = (
    "Employees can roll over up to 5 unused PTO days into the next year. "
    "The retry_timeout default is 3000 seconds. "
    "Staff with 5 years of tenure get an extra 3 PTO days each year."
)


def _make_checker():
    if os.getenv("ANCHOR_CHECKER", "nli").lower() == "coverage":
        return CoverageChecker()
    return NLIChecker(lex_fallback=True, negation_gate=True)


_CHECKER = _make_checker()


@asynccontextmanager
async def lifespan(_app):
    if hasattr(_CHECKER, "_ensure"):
        _CHECKER._ensure()  # warm the model so the first request isn't slow
    yield


app = FastAPI(title="anchor playground", lifespan=lifespan)


class SourceIn(BaseModel):
    id: str
    text: str


class VerifyReq(BaseModel):
    sources: list[SourceIn]
    answer: str


@app.get("/api/health")
def health():
    return {"ok": True, "checker": type(_CHECKER).__name__}


@app.get("/api/samples")
def samples():
    return {"samples": SAMPLES, "answer": SAMPLE_ANSWER}


def _snippet_for(c, sources):
    src = next((s for s in sources if s.id == c.cite_doc), None)
    return src.text[c.cite_start:c.cite_end].strip() if src and c.cite_start >= 0 else ""


@app.post("/api/verify")
def verify(req: VerifyReq):
    srcs = [Source(id=s.id, text=s.text) for s in req.sources]
    claims = Verifier(srcs, _CHECKER).verify(req.answer)
    out, cited = [], []
    for c in claims:
        entry = {"text": c.text, "grounded": c.grounded, "citation": None}
        if c.grounded and c.cite_doc:
            cit = {"n": len(cited) + 1, "doc_id": c.cite_doc,
                   "start": c.cite_start, "end": c.cite_end, "snippet": _snippet_for(c, srcs)}
            cited.append(cit)
            entry["citation"] = cit
        out.append(entry)
    chunks = [{"id": s.id, "chunks": chunk(s.text)} for s in srcs]
    return {"claims": out, "chunks": chunks}


@app.get("/")
def index():
    return FileResponse(_HERE / "static" / "index.html")
