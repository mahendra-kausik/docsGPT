# Plan: "Chat with my docs" — minimal upload MVP

## Context

The deployed project answers questions over one fixed, batch-ingested LangChain corpus.
The user wants a **product-skin demo**: a visitor uploads their own docs and chats over them.
Scope is the **smallest thing that demos it**:

- **Single anonymous session, no auth, no multi-tenancy.** One global Qdrant collection.
- **The upload *replaces* the corpus** (last-uploader-wins). Single-user demo, not a product.

Why it's cheap: the retrieval → agent → API → SSE core is already corpus-agnostic, and
**`build_hybrid_index()` already rebuilds `docs_hybrid` from whatever is in `docs_dense`** — so we
reuse the whole embed/index/search stack. The only new code is: parse uploaded text → chunk →
put it into `docs_dense`, plus one endpoint and a small upload UI.

This is a new build layer under the CLAUDE.md contract: log decisions in `DECISIONS.md`, run the
gate, update `PROGRESS.md`, then stop for approval.

## Design decisions (log in DECISIONS.md at build time)

- **Global-collection replace, no tenancy/auth.** Upload wipes and rebuilds the single
  `docs_dense`/`docs_hybrid` pair. Rationale: smallest demo, free-tier-safe, matches chosen scope.
- **Text-in-JSON upload, not multipart.** Browser `FileReader` reads the file and POSTs
  `{docs: [{filename, text}]}`. Avoids a `python-multipart` dependency and server file handling.
- **Scope to `.md` / `.txt` for MVP.** `chunk_markdown` already handles both. PDF deferred
  (needs `pypdf` server-side or `pdf.js` client-side).
- **Synchronous ingest.** Embed-on-upload blocks the request. Fine for small uploads; no queue.

## Changes

### Backend

1. **`src/retrieval/index.py`** — let the dense builder accept in-memory chunks.
   Extract the embed+upsert loop of `build_index` into `index_chunks(chunks, *, recreate=True)`;
   `build_index` keeps reading `chunks.jsonl` and calls it. No behavior change to existing path.

2. **`src/ingest/upload.py`** (new, small) — `chunks_from_upload(docs) -> list[Chunk]`.
   For each `{filename, text}`: run `chunk_markdown(text, s.chunk_size, s.chunk_overlap, s.min_chunk_chars)`
   and build `Chunk` records exactly as `corpus.build_chunks` does, but with
   `source_url=""`, `source_path=filename`, `version="upload"`, `type="upload"`,
   id = `sha1(f"{filename}::{heading_path}::{idx}")[:16]`. No git clone, no `clean_mdx`.

3. **`src/api/app.py`** — new `POST /upload`:
   - Body: `{docs: [{filename: str, text: str}]}` (Pydantic model).
   - `chunks = chunks_from_upload(...)` → `index_chunks(chunks, recreate=True)`
     → `build_hybrid_index(recreate=True)` (reused verbatim).
   - Return `{files, points}`. Existing `/ask` and `/ask/stream` are untouched — they already
     query the (now user-owned) hybrid collection.

### Frontend

4. **`frontend/src/App.tsx`**:
   - Add a file input (`accept=".md,.txt"`, `multiple`), read each with `FileReader`, POST to
     `${API_URL}/upload`, show "indexing… / N chunks ready" status.
   - **Citation render (line 100):** currently `<a href={c.source_url}>…</a>`; uploaded citations
     have `source_url === ""`. Change to: if `source_url` is truthy render the link, else render
     `<span>{c.heading_path || c.filename}</span>`. (`types.ts` needs no change.)

## Caveats / deferred (mark with `ponytail:` comments where they cut a corner)

- **`recreate=True` wipes the curated LangChain corpus.** Do NOT deploy this pointed at the public
  demo, or any visitor nukes your corpus. Keep upload mode local, or gate it behind a flag.
  Upgrade path (deferred): a separate `docs_user` collection + a per-request collection param
  threaded through `HybridRetriever` in `src/retrieval/search.py` (reopens the hardcoded-collection
  assumption — out of MVP scope).
- **No auth, no multi-tenancy, no storage quota/TTL.** Single-user demo only.
- **PDF/DOCX unsupported.** `.md`/`.txt` only.

## Verification (end-to-end, not just unit)

1. Start backend (`uvicorn src.api.app:app`) + frontend (`vite`).
2. In the UI, upload a small `.md` with a distinctive fact not in the LangChain docs.
3. Confirm `GET /ping` now reports the hybrid collection with the new point count.
4. Ask a question answerable only from the uploaded doc; confirm a grounded answer whose citation
   shows the **filename/heading** (not a broken empty link).
5. One runnable self-check — `tests/test_upload.py`: POST `/upload` with a tiny inline doc, assert
   `points > 0`, then POST `/ask` and assert the answer is non-empty and cites the uploaded chunk.
   (assert-based, no fixtures.)
