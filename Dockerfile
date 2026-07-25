FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# install the anchor package + the [app] extra (NLI core deps + fastapi/uvicorn)
COPY pyproject.toml README.md ./
COPY anchor ./anchor
RUN pip install --no-cache-dir .[app]

# the playground app (examples/benchmarks aren't needed at runtime)
COPY playground ./playground

# pre-cache the NLI model so cold starts are fast (~440MB). Set ANCHOR_CHECKER=coverage to skip.
RUN ANCHOR_CHECKER=nli python -c "from anchor import NLIChecker; NLIChecker()._ensure()"

# runtime loads the precached model from the image only - no HuggingFace contact,
# no update-ping warning. (Set AFTER the precache so the build can still download.)
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV ANCHOR_CHECKER=nli
EXPOSE 8000
CMD ["uvicorn", "playground.app:app", "--host", "0.0.0.0", "--port", "8000"]
