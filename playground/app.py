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
import time
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
     "text": "**Accrual.** Employees accrue 15 days of paid time off per year, available on January 1. "
             "New hires receive a prorated amount based on their start date.\n\n"
             "**Rollover.** Up to 5 unused days roll over into the next calendar year. "
             "Days beyond the 5-day cap are forfeited at year end.\n\n"
             "**Requesting time off.** PTO requests must be submitted at least 7 days in advance through "
             "the HR portal. Manager approval is required for any request exceeding 3 consecutive days."},
    {"id": "remote", "name": "Remote Work Policy",
     "text": "**Eligibility.** All full-time employees are eligible to work remotely. "
             "Contractors must have explicit manager approval.\n\n"
             "**Schedule.** Employees may work remotely up to three days per week. "
             "Core hours are 10am to 3pm Eastern, during which staff must be reachable.\n\n"
             "**Equipment.** A stable internet connection of at least 25 Mbps is required. "
             "The company provides a one-time stipend for home-office equipment."},
    {"id": "expense", "name": "Expense Reimbursement",
     "text": "**Travel.** Lodging is reimbursed up to 200 dollars per night. "
             "Airfare must be booked at least 14 days in advance for the lowest fare.\n\n"
             "**Meals.** Meal expenses are capped at 75 dollars per day. Alcohol is not reimbursable.\n\n"
             "**Approval.** All expenses require an itemized receipt and manager approval, "
             "submitted within 30 days of the charge."},
    {"id": "rate", "name": "API Rate Limits",
     "text": "**Limits.** The public API allows 1000 requests per minute per API key. "
             "Burst capacity permits short spikes up to 1500 requests.\n\n"
             "**Exceeding.** Exceeding the limit returns a 429 Too Many Requests status. "
             "The response includes a Retry-After header indicating when to retry.\n\n"
             "**Backoff.** Clients must implement exponential backoff when retrying rate-limited requests."},
    {"id": "auth", "name": "Authentication",
     "text": "**Tokens.** All API requests require a bearer token in the Authorization header. "
             "Tokens are prefixed with sk_.\n\n"
             "**Expiration.** Access tokens expire after 24 hours. Refresh tokens expire after 30 days.\n\n"
             "**Refresh.** Use the refresh token endpoint to obtain a new access token. "
             "Refresh tokens can be revoked at any time from the dashboard."},
    {"id": "retry", "name": "SDK retry_timeout",
     "text": "**Behavior.** The retry_timeout parameter sets how long the client waits before retrying "
             "a failed request.\n\n"
             "**Default.** The default value is 3000, measured in seconds. "
             "The maximum allowed value is 10000 seconds.\n\n"
             "**Recommendation.** For interactive workloads, set retry_timeout to 1000. "
             "For batch jobs, the default is recommended."},
    {"id": "deploy", "name": "Deployment Runbook",
     "text": "**Schedule.** Production deploys happen on Tuesdays and Thursdays between 10am and 2pm Eastern. "
             "Emergency deploys outside this window require VP approval.\n\n"
             "**Review.** A deploy requires approval from two reviewers. "
             "Changes must pass all CI checks before review.\n\n"
             "**Rollback.** Rollback must be initiated within 15 minutes of a failed deploy. "
             "After 15 minutes, a forward-fix is required."},
    {"id": "oncall", "name": "On-call Rotation",
     "text": "**Rotation.** The on-call rotation cycles weekly across the engineering team. "
             "Shifts run Monday to Monday.\n\n"
             "**Response.** The primary on-call must respond to pages within 5 minutes. "
             "The secondary on-call is the backup and steps in if the primary is unreachable.\n\n"
             "**Handoff.** The outgoing on-call writes a handoff document covering open issues "
             "and ongoing incidents."},
    {"id": "pricing", "name": "Product Pricing",
     "text": "**Plans.** The Pro plan costs 49 dollars per user per month. "
             "The Team plan costs 99 dollars per user per month with a minimum of 5 users.\n\n"
             "**Enterprise.** The Enterprise plan requires an annual contract and includes custom SSO. "
             "Contact sales for pricing.\n\n"
             "**Trial.** A 14-day free trial is available for new accounts on the Pro plan. "
             "No credit card is required."},
    {"id": "refund", "name": "Refund Policy",
     "text": "**Eligibility.** Refunds are available within 30 days of purchase. "
             "Refunds are issued to the original payment method.\n\n"
             "**Subscriptions.** Subscriptions canceled mid-cycle are not refunded for the remaining period. "
             "Annual plans can be refunded pro-rated within the first 30 days.\n\n"
             "**Processing.** Refunds take 5 to 10 business days to appear on the customer's statement."},
    {"id": "retention", "name": "Data Retention",
     "text": "**Customer data.** Customer data is retained for 7 years after account closure, "
             "per regulatory requirements.\n\n"
             "**Backups.** Backup data is retained for 90 days. "
             "Backups older than 90 days are permanently deleted.\n\n"
             "**Deletion.** Data deletion requests are processed within 30 days. "
             "Deletion is irreversible once completed."},
    {"id": "incident", "name": "Incident Response",
     "text": "**Severity.** Severity 1 incidents require a response within 15 minutes. "
             "Severity 2 incidents require a response within 1 hour.\n\n"
             "**Command.** The incident commander coordinates the response and is the single point of communication.\n\n"
             "**Postmortem.** A postmortem is required for all severity 1 and 2 incidents, "
             "published within 5 business days."},
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
    t0 = time.perf_counter()
    claims = Verifier(srcs, _CHECKER).verify(req.answer)
    ms = int((time.perf_counter() - t0) * 1000)
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
    return {"claims": out, "chunks": chunks, "ms": ms}


@app.get("/")
def index():
    return FileResponse(_HERE / "static" / "index.html")
