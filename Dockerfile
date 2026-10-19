FROM python:3.11-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.8.11 /uv /uvx /bin/

ENV PATH="/app/.venv/bin:$PATH" \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# dependencies first (cached layer), then code and model
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev --no-install-project

COPY features.py predict.py ./
COPY models/ ./models/

EXPOSE 9696
ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]
