FROM python:3.13-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy FASTEMBED_CACHE_PATH=/opt/models
WORKDIR /build
COPY requirements.txt .
RUN uv venv /opt/venv && uv pip install --python /opt/venv/bin/python -r requirements.txt
COPY scripts/prefetch_model.py scripts/prefetch_model.py
RUN /opt/venv/bin/python scripts/prefetch_model.py

FROM python:3.13-slim AS runtime
RUN useradd --create-home --uid 1000 appuser
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 FASTEMBED_CACHE_PATH=/opt/models PORT=8000
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder --chown=appuser:appuser /opt/models /opt/models
WORKDIR /app
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser data ./data
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 CMD python -m app.healthcheck
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --no-access-log
