FROM python:3.11-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install \
    --no-cache-dir \
    --target=/opt/python \
    -r /tmp/requirements.txt \
    && mkdir -p /runtime/data/processed /runtime/chroma_db

FROM gcr.io/distroless/python3-debian12:nonroot

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/python \
    PORT=8501

WORKDIR /app

COPY --from=builder --chown=nonroot:nonroot /opt/python /opt/python
COPY --chown=nonroot:nonroot config ./config
COPY --chown=nonroot:nonroot demo_data ./demo_data
COPY --chown=nonroot:nonroot scripts ./scripts
COPY --chown=nonroot:nonroot src ./src
COPY --from=builder --chown=nonroot:nonroot /runtime/data ./data
COPY --from=builder --chown=nonroot:nonroot /runtime/chroma_db ./chroma_db

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD ["/usr/bin/python3", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.getenv('PORT', '8501') + '/_stcore/health', timeout=3)"]

ENTRYPOINT ["/usr/bin/python3", "scripts/container_start.py"]
