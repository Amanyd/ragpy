FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Install dependencies using uv.lock
COPY uv.lock pyproject.toml ./
RUN uv sync --frozen --no-install-project --no-dev

# Copy the rest of the application
COPY . /app
RUN uv sync --frozen --no-dev

# Add the uv python environment to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Expose the API port
EXPOSE 8000

# Run FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
