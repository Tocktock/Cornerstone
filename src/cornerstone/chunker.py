"""Chunking utilities for preparing documents for embedding."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence

_WORD_RE = re.compile(r"\w+")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'\(])')

MAX_TOKENS = 500
MIN_TOKENS = 200
DEFAULT_OVERLAP_TOKENS = 50


@dataclass(slots=True)
class Chunk:
    text: str
    heading_path: tuple[str, ...]
    section_title: str | None
    summary: str | None
    language: str | None
    token_count: int
    char_count: int


def _count_tokens(text: str) -> int:
    return len(_WORD_RE.findall(text))


def _normalize_block(text: str) -> str:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _split_sentences(paragraph: str) -> List[str]:
    paragraph = paragraph.strip()
    if not paragraph:
        return []
    sentences = _SENTENCE_SPLIT_RE.split(paragraph)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def _detect_language(text: str) -> str | None:
    if not text:
        return None
    if any("\uac00" <= char <= "\ud7a3" for char in text):
        return "ko"
    if any("\u4e00" <= char <= "\u9fff" for char in text):
        return "zh"
    return "en"


def _summarize_text(text: str, *, max_length: int = 200) -> str | None:
    sentences = _split_sentences(text)
    if not sentences:
        return None
    summary = sentences[0]
    if len(summary) > max_length:
        summary = summary[: max_length - 1].rstrip() + "…"
    return summary


def _normalize_paragraph_lines(lines: Sequence[str]) -> str:
    stripped_lines = [line.rstrip() for line in lines if line.strip()]
    if not stripped_lines:
        return ""
    is_list = all(line.lstrip().startswith(('-', '*', '+', '1.', '2.', '3.')) for line in stripped_lines)
    if is_list:
        return "\n".join(line.strip() for line in stripped_lines)
    return " ".join(line.strip() for line in stripped_lines)


def _split_paragraphs(section_text: str) -> List[str]:
    paragraphs: List[str] = []
    buffer: List[str] = []
    for line in section_text.split("\n"):
        if not line.strip():
            if buffer:
                paragraphs.append(_normalize_paragraph_lines(buffer))
                buffer = []
            continue
        buffer.append(line)
    if buffer:
        paragraphs.append(_normalize_paragraph_lines(buffer))
    return [paragraph for paragraph in paragraphs if paragraph]


def _split_large_paragraph(paragraph: str, max_tokens: int) -> List[str]:
    if _count_tokens(paragraph) <= max_tokens:
        return [paragraph]
    sentences = _split_sentences(paragraph)
    if not sentences:
        words = paragraph.split()
        return [" ".join(words[i : i + max_tokens]) for i in range(0, len(words), max_tokens)]

    chunks: List[str] = []
    current: List[str] = []
    token_count = 0
    for sentence in sentences:
        tokens = _count_tokens(sentence)
        if token_count + tokens > max_tokens and current:
            chunks.append(" ".join(current))
            current = [sentence]
            token_count = tokens
            continue
        current.append(sentence)
        token_count += tokens
        if token_count >= max_tokens:
            chunks.append(" ".join(current))
            current = []
            token_count = 0
    if current:
        chunks.append(" ".join(current))
    return chunks


def _emit_chunk(
    parts: List[str],
    heading_path: tuple[str, ...],
    section_title: str | None,
    *,
    overlap_ratio: float,
    allow_overlap: bool,
    chunks: List[Chunk],
) -> tuple[str, int]:
    text = "\n\n".join(part.strip() for part in parts if part).strip()
    if not text:
        return "", 0
    token_count = _count_tokens(text)
    char_count = len(text)
    summary = _summarize_text(text)
    language = _detect_language(text)
    chunk = Chunk(
        text=text,
        heading_path=heading_path,
        section_title=section_title,
        summary=summary,
        language=language,
        token_count=token_count,
        char_count=char_count,
    )
    chunks.append(chunk)

    if not allow_overlap or token_count < 2 or overlap_ratio <= 0:
        return "", 0

    words = text.split()
    overlap_tokens = max(1, int(len(words) * overlap_ratio))
    if overlap_tokens >= len(words):
        return "", 0
    carry_words = words[-overlap_tokens:]
    carry_text = " ".join(carry_words)
    return carry_text, _count_tokens(carry_text)


def _section_to_chunks(
    paragraphs: Iterable[str],
    *,
    heading_path: tuple[str, ...],
    section_title: str | None,
    max_tokens: int,
    min_tokens: int,
    target_tokens: int,
    overlap_ratio: float,
) -> List[Chunk]:
    chunks: List[Chunk] = []
    current_parts: List[str] = []
    current_tokens = 0
    carry_text = ""
    carry_tokens = 0

    if carry_text:
        current_parts.append(carry_text)
        current_tokens += carry_tokens

    for paragraph in paragraphs:
        if not paragraph:
            continue
        for piece in _split_large_paragraph(paragraph, max_tokens):
            piece_tokens = _count_tokens(piece)
            if piece_tokens == 0:
                continue
            if current_tokens and current_tokens + piece_tokens > max_tokens:
                carry_text, carry_tokens = _emit_chunk(
                    current_parts,
                    heading_path,
                    section_title,
                    overlap_ratio=overlap_ratio,
                    allow_overlap=True,
                    chunks=chunks,
                )
                current_parts = [carry_text] if carry_text else []
                current_tokens = carry_tokens

            current_parts.append(piece)
            current_tokens += piece_tokens

            if current_tokens >= target_tokens or current_tokens >= max_tokens:
                carry_text, carry_tokens = _emit_chunk(
                    current_parts,
                    heading_path,
                    section_title,
                    overlap_ratio=overlap_ratio,
                    allow_overlap=True,
                    chunks=chunks,
                )
                current_parts = [carry_text] if carry_text else []
                current_tokens = carry_tokens

    if current_parts:
        _emit_chunk(
            current_parts,
            heading_path,
            section_title,
            overlap_ratio=overlap_ratio,
            allow_overlap=False,
            chunks=chunks,
        )

    return chunks


def chunk_markdown(
    text: str,
    *,
    max_tokens: int = MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    min_tokens: int | None = None,
) -> List[Chunk]:
    text = _normalize_block(text)
    if not text:
        return []

    lines = text.split("\n")
    heading_stack: List[str] = []
    section_lines: List[str] = []
    section_title: str | None = None
    chunks: List[Chunk] = []

    min_tokens = min_tokens or max(MIN_TOKENS, max_tokens // 2)
    target_tokens = int(max_tokens * 0.75)
    overlap_ratio = min(max(overlap_tokens / max_tokens, 0.0), 0.5)

    def flush_section() -> None:
        if not section_lines:
            return
        section_text = "\n".join(section_lines).strip()
        if not section_text:
            section_lines.clear()
            return
        paragraphs = _split_paragraphs(section_text)
        section_chunks = _section_to_chunks(
            paragraphs,
            heading_path=tuple(heading_stack),
            section_title=section_title,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            target_tokens=max(target_tokens, min_tokens),
            overlap_ratio=overlap_ratio,
        )
        chunks.extend(section_chunks)
        section_lines.clear()

    for line in lines:
        match = _HEADING_RE.match(line)
        if match:
            flush_section()
            level = len(match.group(1))
            title = match.group(2).strip()
            heading_stack = heading_stack[: level - 1] + [title]
            section_title = title
            continue
        section_lines.append(line)

    flush_section()

    if not chunks:
        paragraphs = _split_paragraphs(text)
        chunks = _section_to_chunks(
            paragraphs,
            heading_path=tuple(),
            section_title=None,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            target_tokens=max(target_tokens, min_tokens),
            overlap_ratio=overlap_ratio,
        )

    return chunks


def chunk_plain_text(
    text: str,
    *,
    max_tokens: int = MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    min_tokens: int | None = None,
) -> List[Chunk]:
    text = _normalize_block(text)
    if not text:
        return []
    min_tokens = min_tokens or max(MIN_TOKENS, max_tokens // 2)
    target_tokens = int(max_tokens * 0.75)
    overlap_ratio = min(max(overlap_tokens / max_tokens, 0.0), 0.5)

    paragraphs = _split_paragraphs(text)
    return _section_to_chunks(
        paragraphs,
        heading_path=tuple(),
        section_title=None,
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        target_tokens=max(target_tokens, min_tokens),
        overlap_ratio=overlap_ratio,
    )


def chunk_text(
    text: str,
    *,
    content_type: str | None = None,
    max_tokens: int = MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    min_tokens: int | None = None,
) -> List[Chunk]:
    text = text.strip()
    if not text:
        return []

    if content_type and "markdown" in content_type:
        return chunk_markdown(text, max_tokens=max_tokens, overlap_tokens=overlap_tokens, min_tokens=min_tokens)

    if text.lstrip().startswith("#"):
        return chunk_markdown(text, max_tokens=max_tokens, overlap_tokens=overlap_tokens, min_tokens=min_tokens)

    return chunk_plain_text(text, max_tokens=max_tokens, overlap_tokens=overlap_tokens, min_tokens=min_tokens)
