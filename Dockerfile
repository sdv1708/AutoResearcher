# ---- base image ---------------------------------------------------
FROM python:3.10-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.8.2

# TODO(cloud): If you need system libs (e.g. libgomp for FAISS‑GPU),
# add `apt-get install` right here.

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl git && \
    rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# ---- production image --------------------------------------------
FROM base AS prod
WORKDIR /app

COPY pyproject.toml poetry.lock* /app/
RUN poetry install --no-dev --only main --no-root

COPY src/ /app/src/

CMD ["poetry", "run", "uvicorn", "autoresearcher.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
