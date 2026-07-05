"""Fetch solved LangChain Forum topics -> gold-eval seeds (Layer 1b).

Context: GitHub Discussions migrated to forum.langchain.com (Discourse) in
mid-2025 (D-018). 'Solved' topics carry an accepted answer -> natural relevance
labels (D-003). We use them as GOLD-EVAL SEEDS ONLY (D-019): the real user
question becomes an eval item (attributed by URL) whose gold *target* is the MIT
docs chunk that answers it (mapped at Layer 3). Forum answer text is NOT added to
the retrieval corpus (kept MIT-clean) and is retained only in the git-ignored raw
cache for local mapping.

Respects robots.txt (uses /c/ and /t/{id}.json; never /search). Public, no auth.

Run:  ./tasks.ps1 ingest-forum
"""

from __future__ import annotations

import argparse
import html
import json
import re
import time
import urllib.request

from src.config import PROJECT_ROOT, ForumConfig, get_settings


def get_json(url: str) -> dict:
    """GET a Discourse JSON endpoint."""
    req = urllib.request.Request(url, headers={"User-Agent": "docsgpt-agent"})
    with urllib.request.urlopen(req, timeout=45) as resp:
        return json.loads(resp.read())


def html_to_text(cooked: str) -> str:
    """Convert Discourse 'cooked' HTML to plain text, preserving code blocks."""
    s = cooked

    def _code_block(m: re.Match) -> str:
        inner = html.unescape(re.sub(r"<[^>]+>", "", m.group(1)))
        return "\n```\n" + inner.strip("\n") + "\n```\n"

    def _inline_code(m: re.Match) -> str:
        return "`" + html.unescape(re.sub(r"<[^>]+>", "", m.group(1))) + "`"

    s = re.sub(r"<pre[^>]*>\s*<code[^>]*>(.*?)</code>\s*</pre>", _code_block, s, flags=re.DOTALL)
    s = re.sub(r"<code[^>]*>(.*?)</code>", _inline_code, s, flags=re.DOTALL)
    s = re.sub(r"<br\s*/?>", "\n", s)
    s = re.sub(r"<li[^>]*>", "- ", s)
    s = re.sub(r"</(p|li|ul|ol|h[1-6]|blockquote|div|tr)>", "\n", s)
    s = re.sub(
        r'<a\s+[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
        lambda m: f"{re.sub(r'<[^>]+>', '', m.group(2))} ({m.group(1)})",
        s,
        flags=re.DOTALL,
    )
    s = re.sub(r"<[^>]+>", "", s)
    s = html.unescape(s)
    return re.sub(r"\n{3,}", "\n\n", s).strip()


def resolve_categories(cfg: ForumConfig) -> list[tuple[str, int, str]]:
    """Map configured display names to (slug, id, name) via categories.json."""
    data = get_json(f"{cfg.base_url}/categories.json")
    cats = data["category_list"]["categories"]
    wanted = set(cfg.categories)
    return [(c["slug"], c["id"], c["name"]) for c in cats if c["name"] in wanted]


def iter_solved_topic_ids(cfg: ForumConfig, slug: str, cat_id: int) -> list[int]:
    """Page a category listing, returning ids of topics with an accepted answer."""
    ids: list[int] = []
    for page in range(cfg.max_pages_per_category):
        data = get_json(f"{cfg.base_url}/c/{slug}/{cat_id}.json?page={page}")
        topics = data.get("topic_list", {}).get("topics", [])
        if not topics:
            break
        ids.extend(t["id"] for t in topics if t.get("has_accepted_answer"))
        if len(topics) < 30:  # last page
            break
        time.sleep(cfg.request_delay_s)
    return ids


def fetch_topic(cfg: ForumConfig, topic_id: int, *, force: bool = False) -> dict:
    """Fetch (and disk-cache) a topic's full JSON."""
    cache = PROJECT_ROOT / cfg.raw_dir / "topics" / f"{topic_id}.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists() and not force:
        return json.loads(cache.read_text(encoding="utf-8"))
    data = get_json(f"{cfg.base_url}/t/{topic_id}.json")
    cache.write_text(json.dumps(data), encoding="utf-8")
    time.sleep(cfg.request_delay_s)
    return data


def parse_topic(tj: dict, category: str, base_url: str) -> dict | None:
    """Extract the question + accepted answer from a topic JSON."""
    posts = tj.get("post_stream", {}).get("posts", [])
    if not posts:
        return None
    accepted = next((p for p in posts if p.get("accepted_answer")), None)
    if accepted is None:  # accepted post not in the loaded stream (rare, long thread)
        return None
    slug, tid = tj["slug"], tj["id"]
    topic_url = f"{base_url}/t/{slug}/{tid}"
    tags = [t["name"] if isinstance(t, dict) else t for t in tj.get("tags", [])]
    return {
        "id": tid,
        "title": tj.get("title", ""),
        "url": topic_url,
        "category": category,
        "tags": tags,
        "created_at": tj.get("created_at", ""),
        "question": html_to_text(posts[0].get("cooked", "")),
        "accepted_answer_post": accepted.get("post_number"),
        "accepted_answer_url": f"{topic_url}/{accepted.get('post_number')}",
        # Kept for local Layer-3 mapping; STRIPPED from the committed seeds file (D-019).
        "accepted_answer": html_to_text(accepted.get("cooked", "")),
    }


# Fields written to the committed gold-seeds file (no forum answer body — D-019).
_SEED_PUBLIC_FIELDS = (
    "id",
    "title",
    "url",
    "category",
    "tags",
    "created_at",
    "question",
    "accepted_answer_post",
    "accepted_answer_url",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch solved LangChain Forum topics.")
    parser.add_argument("--force", action="store_true", help="ignore the topic cache and refetch")
    args = parser.parse_args(argv)

    settings = get_settings()
    cfg = settings.forum
    categories = resolve_categories(cfg)
    if not categories:
        print(f"ERROR: none of {cfg.categories} found among forum categories.")
        return 2

    seeds: list[dict] = []
    for slug, cat_id, name in categories:
        ids = iter_solved_topic_ids(cfg, slug, cat_id)
        print(f"{name}: {len(ids)} solved topics")
        for tid in ids:
            seed = parse_topic(fetch_topic(cfg, tid, force=args.force), name, cfg.base_url)
            if seed:
                seeds.append(seed)

    out_path = (PROJECT_ROOT / cfg.seeds_jsonl).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for s in seeds:
            public = {k: s[k] for k in _SEED_PUBLIC_FIELDS}
            fh.write(json.dumps(public, ensure_ascii=False) + "\n")

    manifest = {
        "source": cfg.base_url,
        "categories": [c[2] for c in categories],
        "n_seeds": len(seeds),
        "note": "gold-eval seeds only; forum answers not in retrieval corpus (D-019)",
    }
    (out_path.parent / "forum_seeds_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("\n=== forum gold seeds built ===")
    print(f"  seeds: {len(seeds)}  -> {out_path}")
    print(f"  raw topic cache (git-ignored): {(PROJECT_ROOT / cfg.raw_dir).resolve()}")
    print("\n=== spot-check (3 seeds) ===")
    for s in seeds[:3]:
        print(f"\n[#{s['id']}] {s['title'][:70]}  ({s['category']})")
        print(f"  {s['url']}  tags={s['tags']}")
        print(f"  Q: {s['question'][:150].strip().replace(chr(10), ' ')}")
        print(f"  accepted answer: {s['accepted_answer_url']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
