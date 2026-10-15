FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/models/huggingface \
    FASTEMBED_CACHE_PATH=/models/fastembed \
    PATH="/app/.venv/bin:$PATH"

RUN pip install --no-cache-dir uv

WORKDIR /app

# Dependencies first, so rebuilds are fast when only code changes.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .

EXPOSE 8001 8501

# The API. The UI service overrides this command in docker-compose.yml.
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8001"]