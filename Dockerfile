FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8501

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY config ./config
COPY demo_data ./demo_data
COPY scripts ./scripts
COPY src ./src

RUN groupadd --system graphmind \
    && useradd --system --gid graphmind --create-home graphmind \
    && mkdir -p /app/data/processed /app/chroma_db \
    && chown -R graphmind:graphmind /app

USER graphmind

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.getenv('PORT', '8501') + '/_stcore/health', timeout=3)"

ENTRYPOINT ["python", "scripts/container_start.py"]
