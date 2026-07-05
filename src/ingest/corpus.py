"""Build the durable chunked corpus from the langchain-ai/docs repo (Layer 1a).

Pipeline: shallow-clone the MIT-licensed docs repo (pin its SHA) -> select the
Python-focused OSS slice by glob -> clean each .mdx -> structure-aware chunk ->
write data/corpus/chunks.jsonl (the durable source of truth, D-004) plus a
manifest recording the exact SHA + config for reproducibility.

Run:  ./tasks.ps1 ingest      (or:  python -m src.ingest.corpus [--force])
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from src.config import PROJECT_ROOT, CorpusConfig, get_settings

from .chunker import chunk_markdown
from .mdx import clean_mdx
from .models import Chunk


def clone_repo(cfg: CorpusConfig, *, force: bool = False) -> tuple[Path, str]:
    """Shallow-clone the docs repo into raw_dir (reused if present); return (dir, short_sha)."""
    raw_dir = (PROJECT_ROOT / cfg.raw_dir).resolve()
    if force and raw_dir.exists():
        import shutil

        shutil.rmtree(raw_dir)
    if not (raw_dir / ".git").exists():
        raw_dir.parent.mkdir(parents=True, exist_ok=True)
        print(f"Cloning {cfg.repo} @ {cfg.ref} -> {raw_dir} ...")
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", cfg.ref, cfg.repo, str(raw_dir)],
            check=True,
        )
    else:
        print(f"Reusing existing clone at {raw_dir}")
    sha = subprocess.run(
        ["git", "-C", str(raw_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return raw_dir, sha[:7]


def select_files(repo_dir: Path, cfg: CorpusConfig) -> list[Path]:
    """Return the sorted, deduped set of .mdx/.md files matching include_globs."""
    seen: set[Path] = set()
    for pattern in cfg.include_globs:
        glob = pattern + "/*" if pattern.endswith("/**") else pattern
        for p in repo_dir.glob(glob):
            if p.is_file() and p.suffix.lower() in {".mdx", ".md"}:
                seen.add(p)
    return sorted(seen)


def path_to_url(rel_path: str, base_url: str) -> str:
    """Map a repo path (src/oss/python/x/index.mdx) to a public docs URL for citations."""
    url_path = rel_path
    for prefix in ("src/",):
        if url_path.startswith(prefix):
            url_path = url_path[len(prefix) :]
    url_path = url_path.rsplit(".", 1)[0]  # drop extension
    if url_path.endswith("/index"):
        url_path = url_path[: -len("/index")]
    return f"{base_url.rstrip('/')}/{url_path}"


def _section(rel_path: str) -> str:
    """oss subtree label, e.g. 'python' from src/oss/python/..."""
    parts = rel_path.split("/")
    return parts[2] if len(parts) > 2 and parts[0] == "src" and parts[1] == "oss" else ""


def build_chunks(repo_dir: Path, sha: str, cfg: CorpusConfig, settings) -> list[Chunk]:
    """Clean + chunk every selected doc into typed Chunk records."""
    snippets_root = repo_dir / "src"
    files = select_files(repo_dir, cfg)
    print(f"Selected {len(files)} doc files.")
    chunks: list[Chunk] = []
    for path in files:
        rel = path.relative_to(repo_dir).as_posix()
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
            meta, cleaned = clean_mdx(raw, snippets_root=snippets_root)
        except Exception as exc:  # noqa: BLE001 - skip a bad file, keep the corpus building
            print(f"  ! skipped {rel}: {exc}", file=sys.stderr)
            continue
        url = path_to_url(rel, cfg.docs_base_url)
        title = str(meta.get("title", "")).strip()
        section = _section(rel)
        page_chunks = chunk_markdown(
            cleaned, settings.chunk_size, settings.chunk_overlap, settings.min_chunk_chars
        )
        for idx, ch in enumerate(page_chunks):
            cid = hashlib.sha1(f"{rel}::{ch['heading_path']}::{idx}".encode()).hexdigest()[:16]
            chunks.append(
                Chunk(
                    id=cid,
                    text=ch["text"],
                    source_url=url,
                    heading_path=ch["heading_path"],
                    version=sha,
                    type="doc",
                    title=title,
                    section=section,
                    source_path=rel,
                    n_chars=len(ch["text"]),
                    has_code=ch["has_code"],
                )
            )
    return chunks


def write_jsonl(chunks: list[Chunk], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(c.model_dump_json() + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the chunked docs corpus.")
    parser.add_argument("--force", action="store_true", help="re-clone the docs repo from scratch")
    args = parser.parse_args(argv)

    settings = get_settings()
    cfg = settings.corpus
    repo_dir, sha = clone_repo(cfg, force=args.force)
    chunks = build_chunks(repo_dir, sha, cfg, settings)

    out_path = (PROJECT_ROOT / settings.corpus_jsonl).resolve()
    write_jsonl(chunks, out_path)

    # Manifest: exactly which snapshot + config produced this corpus (reproducibility).
    manifest = {
        "repo": cfg.repo,
        "ref": cfg.ref,
        "sha": sha,
        "include_globs": cfg.include_globs,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "n_chunks": len(chunks),
        "n_files": len({c.source_path for c in chunks}),
    }
    (out_path.parent / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # --- Gate summary + spot-check ---
    n_code = sum(1 for c in chunks if c.has_code)
    avg = round(sum(c.n_chars for c in chunks) / max(len(chunks), 1))
    by_section: dict[str, int] = {}
    for c in chunks:
        by_section[c.section] = by_section.get(c.section, 0) + 1
    print("\n=== corpus built ===")
    print(f"  chunks: {len(chunks)}  files: {manifest['n_files']}  sha: {sha}")
    print(f"  with code: {n_code}  avg chars: {avg}")
    print(f"  by section: {by_section}")
    print(f"  -> {out_path}")
    print("\n=== spot-check (5 chunks) ===")
    step = max(len(chunks) // 5, 1)
    for c in chunks[:: step][:5]:
        print(f"\n[{c.id}] {c.section} | {c.heading_path}")
        print(f"  url: {c.source_url}")
        preview = c.text[:280].replace("\n", "\n  ")
        print(f"  {preview}{'...' if len(c.text) > 280 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
