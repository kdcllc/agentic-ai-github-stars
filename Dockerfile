FROM python:3.13-slim

# Install system dependencies including curl for health checks
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Install uv using the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy the uv.lock and pyproject.toml files
COPY pyproject.toml uv.lock ./

# Add streamlit and other dependencies to the project dependencies
RUN echo 'streamlit = ">=1.28.0"' >> pyproject.toml && \
    echo 'pandas = "*"' >> pyproject.toml && \
    echo 'openai = ">=1.0.0"' >> pyproject.toml && \
    echo 'azure-identity = "*"' >> pyproject.toml

# Install dependencies with uv in virtual environment mode
RUN uv sync --frozen --no-cache

# Copy the rest of the application
COPY . .

# Expose the Streamlit port and health check port
EXPOSE 8501 8000

# Set environment variables for Streamlit
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8501 \
    HEALTHCHECK_PORT=8000 \
    EMBEDDING_PROVIDER=sentence_transformer

# Add health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -fs http://localhost:8000/health || exit 1

# Run the Streamlit application using the uv virtual environment
CMD ["uv", "run", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
