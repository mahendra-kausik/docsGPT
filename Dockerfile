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

# Bake the retrieval models into the image so the first request does NO network
# download. On scale-to-zero Cloud Run a cold /ask otherwise stalls fetching bge-small
# + Qdrant/bm25 from HF (unauthenticated rate limit) and blows past the 600s request
# cap -> 504. Cache to /opt (NOT /tmp: Cloud Run mounts /tmp as runtime tmpfs, which
# would hide anything baked there). Keys must match src/config.py defaults.
ENV HF_HOME=/opt/models/hf \
    FASTEMBED_CACHE_PATH=/opt/models/fastembed
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')" \
 && python -c "from fastembed import SparseTextEmbedding; SparseTextEmbedding('Qdrant/bm25')"
# Now that both models are on disk, forbid runtime HF lookups so a slow/failed etag
# check can never re-introduce the stall; the baked files satisfy every load offline.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY . .

# Cloud Run injects $PORT (default 8080). Shell form so it expands at runtime.
ENV PORT=8080
CMD ["sh", "-c", "uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
