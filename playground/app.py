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

from anchor import CoverageChecker, NLIChecker, Source, annotate

_HERE = Path(__file__).parent

SAMPLES = [
    {"id": "PTO-Policy", "name": "PTO Policy",
     "text": "Employees get 15 days of PTO per year. Up to 5 unused days roll over "
             "into the next year. Requests must be submitted 7 days in advance."},
    {"id": "SDK-Docs", "name": "API Docs",
     "text": "The retry_timeout parameter sets how long the client waits before "
             "retrying a failed request. The default value is 3000, measured in seconds."},
    {"id": "Remote-Work", "name": "Remote Work Handbook",
     "text": "Employees may work remotely up to three days per week. A monthly stipend "
             "of 75 dollars is provided for home internet. Travel to the office is "
             "required once per quarter."},
]
SAMPLE_ANSWER = (
    "You can roll over up to 5 unused days into next year. "
    "To request time off, submit your request at least 7 days in advance. "
    "Employees with 5+ years of tenure qualify for 3 bonus vacation days annually."
)


def _make_checker():
    return CoverageChecker() if os.getenv("ANCHOR_CHECKER", "nli").lower() == "coverage" else NLIChecker()


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


@app.post("/api/verify")
def verify(req: VerifyReq):
    srcs = [Source(id=s.id, text=s.text) for s in req.sources]
    ans = annotate(req.answer, srcs, _CHECKER)
    return {
        "text": ans.text,
        "citations": [{"n": c.n, "doc_id": c.doc_id, "start": c.start, "end": c.end,
                       "snippet": c.snippet} for c in ans.citations],
        "flags": ans.flags,
    }


@app.get("/")
def index():
    return FileResponse(_HERE / "static" / "index.html")
