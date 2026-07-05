"""Structure-aware chunking of cleaned Markdown (Layer 1).

Why this design: the plan requires chunks that (a) NEVER split a fenced code
block and (b) carry a heading breadcrumb. We first split the doc into atomic
blocks (heading / code / text) with explicit fence tracking, then greedily pack
blocks up to `chunk_size` characters, flushing on headings and on size. Overlap
is carried only from trailing *prose* (never across a code block), so code stays
whole and context bleeds between prose chunks the way retrievers like.
"""

from __future__ import annotations

import re

from .mdx import FENCE_RE, HEADING_RE

_HEADING_LINE_RE = re.compile(r"^\s*#{1,6}\s.*$", re.MULTILINE)
_IMAGE_LINK_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")


def _has_substance(text: str, min_chars: int) -> bool:
    """True unless the chunk is essentially empty (heading-only / lone image link)."""
    if "```" in text:
        return True  # code always counts as content
    body = _IMAGE_LINK_RE.sub("", _HEADING_LINE_RE.sub("", text))
    return len(body.strip()) >= min_chars


def _blocks(body: str) -> list[dict]:
    """Split body into ordered atomic blocks: heading | code | text."""
    lines = body.split("\n")
    blocks: list[dict] = []
    buf: list[str] = []
    i = 0

    def flush_text() -> None:
        if any(line.strip() for line in buf):
            blocks.append({"kind": "text", "text": "\n".join(buf).strip("\n")})
        buf.clear()

    while i < len(lines):
        line = lines[i]
        fm = FENCE_RE.match(line)
        if fm:  # opening fence -> consume through the matching close
            flush_text()
            marker = fm.group(2)[0]
            code = [line]
            i += 1
            while i < len(lines):
                l2 = lines[i]
                code.append(l2)
                fm2 = FENCE_RE.match(l2)
                i += 1
                if fm2 and fm2.group(2)[0] == marker and fm2.group(3).strip() == "":
                    break
            blocks.append({"kind": "code", "text": "\n".join(code)})
            continue
        hm = HEADING_RE.match(line)
        if hm:
            flush_text()
            blocks.append(
                {
                    "kind": "heading",
                    "level": len(hm.group(1)),
                    "title": hm.group(2).strip(),
                    "text": line,
                }
            )
            i += 1
            continue
        if line.strip() == "":
            flush_text()
            i += 1
            continue
        buf.append(line)
        i += 1
    flush_text()
    return blocks


def _overlap_tail(cur: list[dict], n: int) -> str:
    """Last ~n chars of the trailing prose block; empty if the chunk ended in code."""
    if n <= 0:
        return ""
    for blk in reversed(cur):
        if blk["kind"] == "code":
            return ""
        if blk["kind"] == "text":
            tail = blk["text"][-n:]
            sp = tail.find(" ")
            return tail[sp + 1 :] if 0 < sp < len(tail) - 1 else tail
    return ""


def chunk_markdown(
    body: str, chunk_size: int, chunk_overlap: int, min_chunk_chars: int = 0
) -> list[dict]:
    """Return ordered chunks: [{text, heading_path, has_code}].

    A code block larger than `chunk_size` becomes its own oversize chunk rather
    than being split — keeping code intact is a hard requirement of the gate.
    Chunks with no real content (heading-only / lone image) are dropped when
    `min_chunk_chars` > 0.
    """
    blocks = _blocks(body)
    chunks: list[dict] = []
    stack: list[tuple[int, str]] = []
    cur: list[dict] = []
    cur_len = 0
    cur_path = ""
    pending_heading: str | None = None

    def heading_path() -> str:
        return " > ".join(title for _, title in stack)

    def flush() -> None:
        nonlocal cur, cur_len
        text = "\n\n".join(b["text"] for b in cur).strip()
        if text and _has_substance(text, min_chunk_chars):
            chunks.append(
                {
                    "text": text,
                    "heading_path": cur_path,
                    "has_code": any(b["kind"] == "code" for b in cur),
                }
            )
        cur = []
        cur_len = 0

    for b in blocks:
        if b["kind"] == "heading":
            flush()
            lvl = b["level"]
            while stack and stack[-1][0] >= lvl:
                stack.pop()
            stack.append((lvl, b["title"]))
            cur_path = heading_path()
            pending_heading = b["text"]
            continue

        if not cur and pending_heading is not None:  # lead a fresh chunk with its heading
            cur.append({"kind": "text", "text": pending_heading})
            cur_len += len(pending_heading)
            pending_heading = None

        blen = len(b["text"])
        if cur and cur_len + blen > chunk_size:
            tail = _overlap_tail(cur, chunk_overlap)
            flush()
            if tail:
                cur.append({"kind": "text", "text": tail})
                cur_len += len(tail)
        cur.append(b)
        cur_len += blen

    flush()
    return chunks
