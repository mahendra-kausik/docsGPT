"""Unit tests for the MDX cleaner + structure-aware chunker (Layer 1a gate)."""

from src.ingest.chunker import chunk_markdown
from src.ingest.mdx import clean_mdx

SAMPLE = """\
---
title: "My Page"
description: "a description"
---

import Foo from '/snippets/x.mdx'

Intro paragraph before any heading.

## Setup

Install it first.

```python
import os
# this is a python comment, not a heading
os.environ["X"] = "1"
```

### Details

<Note>
Keep this note text.
</Note>

More details here.
"""


def _clean_and_chunk(text, size=1000, overlap=150):
    _, body = clean_mdx(text)
    return clean_mdx(text)[0], chunk_markdown(body, size, overlap)


def test_frontmatter_extracted_and_stripped():
    meta, body = clean_mdx(SAMPLE)
    assert meta["title"] == "My Page"
    assert "description:" not in body  # frontmatter block removed from body


def test_mdx_import_and_jsx_tags_stripped_but_text_kept():
    _, body = clean_mdx(SAMPLE)
    assert "import Foo" not in body        # MDX import line removed
    assert "<Note>" not in body and "</Note>" not in body  # JSX tags removed
    assert "Keep this note text." in body  # ...but inner prose survives


def test_code_block_intact_and_hash_lines_not_headings():
    meta, chunks = _clean_and_chunk(SAMPLE)
    code_chunks = [c for c in chunks if c["has_code"]]
    assert code_chunks, "expected a chunk flagged as containing code"
    code = code_chunks[0]
    # The whole fenced block survives, including the '#'-comment that looks like a heading.
    assert "import os" in code["text"]
    assert "# this is a python comment, not a heading" in code["text"]
    # A '#' line inside a fence must never leak into a heading breadcrumb.
    for c in chunks:
        assert "python comment" not in c["heading_path"]


def test_heading_path_breadcrumb():
    _, chunks = _clean_and_chunk(SAMPLE)
    paths = {c["heading_path"] for c in chunks}
    assert "Setup" in paths                    # content under ## Setup
    assert "Setup > Details" in paths          # nested ### Details


DIRECTIVE_SAMPLE = """\
## Context

:::python
Use the Python context object.

```python
ctx = Context()
```
:::

:::js
Use the JS context object.

```typescript
const ctx = new Context();
```
:::

:::note
Remember this note.
:::
"""


def test_conditional_content_keeps_python_drops_js():
    _, body = clean_mdx(DIRECTIVE_SAMPLE)
    assert "Use the Python context object." in body   # :::python kept
    assert "ctx = Context()" in body
    assert "Use the JS context object." not in body    # :::js dropped
    assert "new Context()" not in body
    assert "typescript" not in body
    assert "Remember this note." in body               # admonition content kept
    assert ":::" not in body                           # all markers stripped


def test_base64_image_stripped():
    blob = "A" * 5000
    doc = f'## Pic\n\n<p>\n<img src="data:image/png;base64,{blob}" />\n</p>\n\nCaption text.\n'
    _, body = clean_mdx(doc)
    assert "base64" not in body
    assert blob not in body
    assert "Caption text." in body


def test_nested_and_inline_directives():
    doc = (
        "::::::js\nJS only content here.\n::::::\n\n"
        ":::caution This API is deprecated.\n:::\n\n"
        ":::python\nPython content kept.\n:::\n"
    )
    _, body = clean_mdx(doc)
    assert "JS only content here." not in body        # 6-colon js block dropped
    assert "This API is deprecated." in body          # inline admonition content kept
    assert "Python content kept." in body
    assert ":::" not in body


def test_empty_heading_only_chunks_dropped():
    doc = "## Orphan heading no body\n\n## Real section\n\nActual content lives here in prose.\n"
    chunks = chunk_markdown(doc, chunk_size=1000, chunk_overlap=0, min_chunk_chars=24)
    paths = [c["heading_path"] for c in chunks]
    assert "Orphan heading no body" not in paths  # heading-only chunk dropped
    assert "Real section" in paths                      # section with prose kept


def test_oversize_code_block_is_not_split():
    big = "```python\n" + "\n".join(f"x{i} = {i}" for i in range(400)) + "\n```\n"
    doc = "## Big\n\n" + big
    chunks = chunk_markdown(doc, chunk_size=200, chunk_overlap=50)
    code_chunks = [c for c in chunks if c["has_code"]]
    assert len(code_chunks) == 1                # a single oversize chunk, not fragments
    assert code_chunks[0]["text"].count("```") == 2  # exactly one intact fence pair
