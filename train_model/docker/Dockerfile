FROM tensorflow/tensorflow:2.17.0
# version for cpu


COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_SYSTEM_PYTHON=1
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

ENV PATH="/app/.venv/bin:$PATH"
