FROM python:3.12-slim

WORKDIR /app

# Tarik perkakas dasar OS
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Pakai uv untuk ngebut narik paket
RUN pip install uv

# Tumpahkan semua amunisi kode sumber
COPY . .

# Bangun lingkungan kedap suara (venv) dan instalasi dependensi
RUN uv venv
RUN uv pip install -r requirements.txt || true
# Cadangan apabila daftar requirements belum direkam rapi
RUN uv pip install fastapi uvicorn streamlit pydantic setfit sentence-transformers pandas scikit-learn litellm duckdb python-multipart python-dotenv PySastrawi openai-whisper

# Setel variabel lingkungan
ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:$PATH"

# Buka gerbang port untuk API backend (8000) dan UI Dasbor (8501)
EXPOSE 8000 8501

# Komando hidup standar (bakal ditimpa oleh docker-compose)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
