"""Turn the PDFs in backend/data/ into embedded chunks in the Chroma collection.

Chunking is two-stage and structure-aware:

1. If a note carries markdown headers, split on them so no chunk crosses a
   header boundary, and record the header path ("Databases > Indexing").
2. Split each section (or the whole note, for headerless plain text) with
   RecursiveCharacterTextSplitter, sized in the embedding model's own tokens.

Usage:
    python -m backend.ingest              # ingest every PDF in data/
    python -m backend.ingest --reset      # wipe the collection first
    python -m backend.ingest notes.pdf    # ingest specific files
"""

from __future__ import annotations

import argparse
import functools
import re
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from backend.retrieval import DATA_DIR

# Sized in characters. The Gemini embedding model accepts far more than a chunk
# this size, so there is no cap to apply; ~4 chars/token puts 1200 characters
# at roughly the 300-token target.
CHUNK_CHARS = 1200
OVERLAP_CHARS = 160

# Prefer paragraph breaks, then line breaks, then sentence ends, then words.
# The empty string is the last resort: a hard cut mid-word.
SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""]

# A markdown ATX header: 1-6 '#' followed by whitespace and real text.
# Requiring the space excludes "#1", "#tag", and stray '#' in prose.
HEADER_RE = re.compile(r"^(#{1,6})[ \t]+(\S.*?)[ \t]*#*$", re.MULTILINE)

# How many headers a note needs before we treat it as markdown.
MIN_HEADERS = 2
# If nearly every line is a header it is a table of contents, not prose.
MAX_HEADER_LINE_RATIO = 0.5

SUPPORTED_SUFFIXES = {".pdf", ".md", ".txt", ".markdown"}

# --- repeated header/footer removal -------------------------------------
# Slide decks and papers repeat a banner on every page. It is identical
# everywhere, so it both dilutes each chunk's embedding and pulls all chunks
# toward each other, flattening the ranking.
BOILERPLATE_MIN_PAGES = 4  # too few pages to tell repetition from coincidence
BOILERPLATE_PAGE_RATIO = 0.6  # share of pages a line must appear on
BOILERPLATE_MIN_CHARS = 8  # ignore fragments; too short to judge

# Leading slide/section numbers ("6.10 ", "3-2 ") differ per page while the
# rest of the banner is identical, so they are normalized away before counting.
_LEADING_NUMBER = re.compile(r"^\s*\d+(?:[.\-]\d+)*\s*")
_PAGE_NUMBER_ONLY = re.compile(r"^\s*(page\s*)?\d+(?:\s*/\s*\d+)?\s*$", re.I)


@dataclass
class Section:
    """A run of text under one header path (or the whole note, if headerless)."""

    text: str
    start: int  # character offset into the full document text
    header_path: str  # "Databases > Indexing", or "" for plain text


@dataclass
class Chunk:
    text: str
    page: int
    header_path: str


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------


def read_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Extract (page_number, text) pairs, skipping pages with no extractable text."""
    reader = PdfReader(str(pdf_path))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((page_number, text))
    return pages


def read_text_file(path: Path) -> list[tuple[int, str]]:
    """Read a .md or .txt note as a single pageless unit.

    Page 0 means "this format has no pages"; citations omit the page for these,
    and for markdown the header path carries the location instead.
    """
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    return [(0, text)] if text else []


def read_document(path: Path) -> list[tuple[int, str]]:
    """Read any supported note, with repeated banners stripped."""
    if path.suffix.lower() == ".pdf":
        pages = read_pages(path)
    else:
        pages = read_text_file(path)
    return strip_boilerplate(pages)


def _normalize_line(line: str) -> str:
    """Canonical form of a line for repetition counting."""
    return re.sub(r"\s+", " ", _LEADING_NUMBER.sub("", line.strip())).lower()


def find_boilerplate(pages: list[tuple[int, str]]) -> set[str]:
    """Normalized lines that repeat on most pages, i.e. headers and footers."""
    if len(pages) < BOILERPLATE_MIN_PAGES:
        return set()

    counts: dict[str, int] = {}
    for _, text in pages:
        # Count each distinct line once per page; a line repeated within one
        # page is content, not a header.
        for norm in {_normalize_line(l) for l in text.splitlines()}:
            if len(norm) >= BOILERPLATE_MIN_CHARS:
                counts[norm] = counts.get(norm, 0) + 1

    threshold = max(2, int(len(pages) * BOILERPLATE_PAGE_RATIO))
    return {norm for norm, n in counts.items() if n >= threshold}


def strip_boilerplate(pages: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Drop repeated banners and bare page numbers from every page."""
    banners = find_boilerplate(pages)

    cleaned = []
    for page_number, text in pages:
        kept = [
            line
            for line in text.splitlines()
            if _normalize_line(line) not in banners
            and not _PAGE_NUMBER_ONLY.match(line)
        ]
        body = "\n".join(kept).strip()
        if body:
            cleaned.append((page_number, body))
    return cleaned


def build_document(pages: list[tuple[int, str]]) -> tuple[str, list[int], list[int]]:
    """Join pages into one string, keeping an offset -> page-number index.

    Header sections routinely span page breaks, so stage 1 has to see the whole
    note. The offset index lets every chunk still report the page it started on.
    """
    parts, offsets, numbers = [], [], []
    cursor = 0
    for page_number, text in pages:
        offsets.append(cursor)
        numbers.append(page_number)
        parts.append(text)
        cursor += len(text) + 2  # the "\n\n" join below
    return "\n\n".join(parts), offsets, numbers


def page_at(offset: int, offsets: list[int], numbers: list[int]) -> int:
    """Page number containing `offset`, via binary search over page starts."""
    if not offsets:
        return 0
    return numbers[max(0, bisect_right(offsets, offset) - 1)]


# --------------------------------------------------------------------------
# Stage 1 - markdown header detection and splitting
# --------------------------------------------------------------------------


def looks_like_markdown(text: str) -> bool:
    """True when the note has real header structure worth splitting on.

    Two guards: enough headers to be structure rather than noise, and not so
    many that the "note" is really an outline or table of contents.
    """
    headers = HEADER_RE.findall(text)
    if len(headers) < MIN_HEADERS:
        return False
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    return len(headers) / len(lines) <= MAX_HEADER_LINE_RATIO


def split_by_headers(text: str) -> list[Section]:
    """Split on markdown headers, tracking the nesting path down to each section.

    Returns sections in document order with absolute start offsets, so page
    attribution survives. Text before the first header becomes a section with
    an empty header path.
    """
    matches = list(HEADER_RE.finditer(text))
    if not matches:
        return [Section(text=text, start=0, header_path="")]

    sections: list[Section] = []
    stack: list[tuple[int, str]] = []  # (level, title), outermost first

    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(Section(text=preamble, start=0, header_path=""))

    for i, match in enumerate(matches):
        level = len(match.group(1))
        title = match.group(2).strip()

        # Pop siblings and deeper headers, then push this one.
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))

        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end]

        stripped = body.strip()
        if not stripped:
            # A header with no body of its own (e.g. an H1 above H2s) still
            # contributes its title to the path of the sections below it.
            continue

        sections.append(
            Section(
                text=stripped,
                start=body_start + (len(body) - len(body.lstrip())),
                header_path=" > ".join(title for _, title in stack),
            )
        )

    return sections


# --------------------------------------------------------------------------
# Stage 2 - recursive splitting within each section
# --------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def get_splitter():
    """Recursive splitter, sized in characters.

    Imported lazily: langchain_text_splitters pulls in langchain-core and costs
    ~365 MB of RSS. Chunking only happens while indexing, so the serving path
    should never pay for it.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_CHARS,
        chunk_overlap=OVERLAP_CHARS,
        separators=SEPARATORS,
        add_start_index=True,
        keep_separator=True,
    )


def chunk_document(pages: list[tuple[int, str]]) -> list[Chunk]:
    """Two-stage split of one note into embeddable chunks with metadata."""
    text, offsets, numbers = build_document(pages)
    if not text.strip():
        return []

    sections = (
        split_by_headers(text)
        if looks_like_markdown(text)
        else [Section(text=text, start=0, header_path="")]
    )

    splitter = get_splitter()
    chunks: list[Chunk] = []
    for section in sections:
        # Chunks arrive in document order, so a cursor lets us locate any chunk
        # the splitter could not place and keeps offsets monotonic.
        cursor = 0
        for doc in splitter.create_documents([section.text]):
            body = doc.page_content.strip()
            if not body:
                continue
            start = doc.metadata.get("start_index", -1)
            if start is None or start < 0:
                # add_start_index reports -1 when its exact match fails - the
                # splitter rejoins separators differently from the source. Fall
                # back to a forward search, then to the previous position, so a
                # chunk is never attributed to page 1 by accident.
                found = section.text.find(body[:60], cursor)
                start = found if found >= 0 else cursor
            cursor = max(cursor, start)
            offset = section.start + start
            chunks.append(
                Chunk(
                    text=body,
                    page=page_at(offset, offsets, numbers),
                    header_path=section.header_path,
                )
            )
    return chunks


# --------------------------------------------------------------------------
# Ingestion
# --------------------------------------------------------------------------


def ingest_pdf(pdf_path: Path) -> int:
    """(Re-)ingest one PDF. Thin wrapper kept for backwards compatibility.

    The embed-and-store half of the pipeline lives in `embed_and_store`; this
    delegates so there is only one definition of a stored record. Imported
    inside the function because `embed_and_store` imports this module.
    """
    from backend.embed_and_store import embed_pdf

    return embed_pdf(pdf_path)


def ingest(paths: list[Path] | None = None, reset: bool = False) -> int:
    """Ingest the given PDFs, or every PDF in data/ when none are given."""
    from backend.embed_and_store import embed_all

    return embed_all(paths, reset=reset)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdfs", nargs="*", type=Path, help="specific PDFs to ingest")
    parser.add_argument("--reset", action="store_true", help="clear the collection first")
    args = parser.parse_args()

    paths = [p if p.exists() else DATA_DIR / p.name for p in args.pdfs]
    total = ingest(paths or None, reset=args.reset)
    print(f"Done - {total} chunks indexed.")


if __name__ == "__main__":
    main()
