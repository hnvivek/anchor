FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# library deps, then app deps (app deps pull the library via -r ../requirements.txt)
COPY requirements.txt ./
COPY playground/requirements.txt playground/requirements.txt
RUN pip install --no-cache-dir -r playground/requirements.txt

# the library + the playground app
COPY anchor.py ./
COPY playground ./playground

# Pre-cache the NLI model so cold starts are fast (~440MB into the image layer).
# Set ANCHOR_CHECKER=coverage at runtime to skip the model entirely (tiny tier).
RUN ANCHOR_CHECKER=nli python -c "from anchor import NLIChecker; NLIChecker()._ensure()"

ENV ANCHOR_CHECKER=nli
EXPOSE 8000
CMD ["uvicorn", "playground.app:app", "--host", "0.0.0.0", "--port", "8000"]
