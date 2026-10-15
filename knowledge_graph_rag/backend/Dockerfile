FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/app/.cache/huggingface \
    DATA_DIR=/data

# PDF parsing (unstructured): poppler, tesseract OCR, libmagic, OpenCV runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends \
        poppler-utils tesseract-ocr libmagic1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 app
WORKDIR /app

# uv resolves this dependency set in seconds (pip backtracks for 20+ minutes).
# CPU-only torch keeps the image ~2 GB smaller than the default CUDA build.
COPY --from=ghcr.io/astral-sh/uv:0.9.17 /uv /usr/local/bin/uv
COPY requirements.txt .
RUN uv pip install --system --no-cache \
        --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
        torch==2.14.1+cpu -r requirements.txt

COPY --chown=app:app . .
RUN mkdir -p /data "$HF_HOME" && chown -R app:app /data /home/app/.cache

USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
