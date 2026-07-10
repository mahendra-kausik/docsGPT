# DocsGPT-Agent API image (Layer 8a). Serves the cited-answer agent on Cloud Run.
# CPU-only: no GPU on Cloud Run, so torch comes from the CPU wheel index (D-023) —
# the default PyPI torch is a multi-GB CUDA build we neither need nor can run.
FROM python:3.13-slim

WORKDIR /app

# torch first, from the CPU wheel index, so the pinned torch==2.12.1 resolves to the
# CPU build; requirements.txt then finds it already satisfied (D-023).
RUN pip install --no-cache-dir torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cloud Run injects $PORT (default 8080). Shell form so it expands at runtime.
ENV PORT=8080
CMD ["sh", "-c", "uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
