FROM python:3.12-slim

WORKDIR /app

# Install system dependencies (ffmpeg for ASR audio processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
RUN pip install --no-cache-dir uv

# Copy dependency list first for layer caching
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN uv venv && uv pip install -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:$PATH"

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Default: start FastAPI (docker-compose overrides per service)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
