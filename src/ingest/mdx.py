"""MDX cleaning: turn a raw .mdx doc into clean Markdown text + frontmatter.

Why hand-rolled (not a Markdown/MDX library): the Layer 1 gate demands code
blocks stay intact and headings map to a breadcrumb. That is easiest to
*guarantee* by walking lines with explicit fenced-code tracking, and it keeps
the dependency surface lean (DECISIONS D-015). MDX-isms we strip: YAML
frontmatter, `import ... from '...'` / `export ...` statements, JSX component
tags (keeping their inner text), and `{/* ... */}` comments — but ONLY outside
fenced code, so Python snippets like `import os` are never touched.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

FENCE_RE = re.compile(r"^(\s*)(`{3,}|~{3,})(.*)$")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
# MDX import/export at line start (JS-style `import X from '...'`, distinct from
# Python `import os` which has no `from '...'` and lives inside code anyway).
MDX_IMPORT_RE = re.compile(r"""^\s*import\s+.+\s+from\s+['"][^'"]+['"]\s*;?\s*$""")
MDX_IMPORT_SIDE_RE = re.compile(r"""^\s*import\s+['"][^'"]+['"]\s*;?\s*$""")
MDX_EXPORT_RE = re.compile(r"^\s*export\s+.+$")
JSX_COMMENT_RE = re.compile(r"\{/\*.*?\*/\}", re.DOTALL)
# Component tags start with an uppercase letter (Mintlify: <CodeGroup>, <Note>, <Tab ...>).
JSX_TAG_RE = re.compile(r"</?[A-Z][A-Za-z0-9]*(?:\s[^>]*?)?/?>")
# `import Name from '/snippets/....mdx'` -> map component name to snippet path.
SNIPPET_IMPORT_RE = re.compile(
    r"""^\s*import\s+([A-Za-z0-9_]+)\s+from\s+['"](/?snippets/[^'"]+|/snippets/[^'"]+)['"]"""
)
# Mintlify conditional-content / admonition directives: ':::python' ... ':::'
# Allow 3+ colons (nested levels like ':::::js') and inline content after the name.
DIRECTIVE_OPEN_RE = re.compile(r"^\s*:{3,}([A-Za-z][\w-]*)\b[ \t]*(.*)$")
DIRECTIVE_CLOSE_RE = re.compile(r"^\s*:{3,}\s*$")
# Language variants to DROP for a Python-focused corpus (keep :::python, drop :::js).
_DROP_LANGS = {"js", "javascript", "ts", "typescript"}
# Inline HTML images embed huge base64 data URIs — useless for text retrieval and enormous.
IMG_TAG_RE = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
DATA_URI_RE = re.compile(r"data:[a-zA-Z0-9.+-]+/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=]+")


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split leading `---`-delimited YAML frontmatter from the body.

    Returns (metadata, body). Missing/malformed frontmatter yields ({}, text).
    """
    if not text.startswith("---"):
        return {}, text
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", text, re.DOTALL)
    if not m:
        return {}, text
    try:
        meta = yaml.safe_load(m.group(1)) or {}
        if not isinstance(meta, dict):
            meta = {}
    except yaml.YAMLError:
        meta = {}
    return meta, m.group(2)


def _iter_snippet_imports(body: str) -> dict[str, str]:
    """Collect {ComponentName: snippet_path} from MDX import lines (outside code)."""
    mapping: dict[str, str] = {}
    for line in body.splitlines():
        m = SNIPPET_IMPORT_RE.match(line)
        if m:
            path = m.group(2).lstrip("/")  # 'snippets/...'
            mapping[m.group(1)] = path
    return mapping


def clean_mdx(
    text: str,
    *,
    snippets_root: Path | None = None,
    _depth: int = 0,
    _seen: set[str] | None = None,
) -> tuple[dict, str]:
    """Return (frontmatter_meta, cleaned_markdown_body).

    Best-effort inlines `/snippets/*.mdx` transclusions one/few levels deep
    (recursion-guarded); drops unresolved component tags.
    """
    meta, body = parse_frontmatter(text)
    _seen = _seen if _seen is not None else set()

    snippet_map = _iter_snippet_imports(body) if snippets_root else {}

    out: list[str] = []
    in_fence = False
    fence_marker = ""
    directive_stack: list[str] = []  # 'keep' | 'drop' frames for ::: blocks
    dropping = False

    for line in body.splitlines():
        fm = FENCE_RE.match(line)
        if fm:
            marker = fm.group(2)
            if not in_fence:
                in_fence, fence_marker = True, marker[0]
                if not dropping:
                    out.append(line)
                continue
            # closing fence: same marker char, no trailing info
            if marker[0] == fence_marker and fm.group(3).strip() == "":
                in_fence = False
                if not dropping:
                    out.append(line)
                continue
            if not dropping:
                out.append(line)
            continue

        if in_fence:
            if not dropping:
                out.append(line)  # never touch code
            continue

        # --- ::: conditional-content / admonition directives ---
        if DIRECTIVE_CLOSE_RE.match(line):
            if directive_stack:
                directive_stack.pop()
            dropping = "drop" in directive_stack
            continue
        dopen = DIRECTIVE_OPEN_RE.match(line)
        if dopen:
            name = dopen.group(1).lower()
            mode = "drop" if (dropping or name in _DROP_LANGS) else "keep"
            directive_stack.append(mode)
            dropping = "drop" in directive_stack
            trailing = dopen.group(2).strip()  # inline content, e.g. ':::caution Deprecated.'
            if trailing and not dropping:
                out.append(trailing)
            continue
        if dropping:
            continue

        # --- outside code: strip MDX-isms ---
        if (
            MDX_IMPORT_RE.match(line)
            or MDX_IMPORT_SIDE_RE.match(line)
            or MDX_EXPORT_RE.match(line)
        ):
            continue

        stripped = line.strip()
        # A line that is only a snippet-component usage -> inline it.
        comp = re.match(r"^<([A-Z][A-Za-z0-9]*)\s*/?>$", stripped)
        if comp and comp.group(1) in snippet_map and snippets_root and _depth < 3:
            spath = snippets_root / snippet_map[comp.group(1)]
            if spath.exists() and str(spath) not in _seen:
                _seen.add(str(spath))
                _, inlined = clean_mdx(
                    spath.read_text(encoding="utf-8", errors="replace"),
                    snippets_root=snippets_root,
                    _depth=_depth + 1,
                    _seen=_seen,
                )
                out.append(inlined)
            continue

        line = IMG_TAG_RE.sub("", line)       # drop <img> (often megabyte base64 data URIs)
        line = DATA_URI_RE.sub("", line)      # safety net for stray base64 blobs
        line = JSX_COMMENT_RE.sub("", line)
        line = JSX_TAG_RE.sub("", line)
        out.append(line)

    cleaned = "\n".join(out)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip() + "\n"
    return meta, cleaned
