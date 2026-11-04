"""Parse FB2 books and ingest structured chunks into PostgreSQL."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from uuid import NAMESPACE_URL, uuid5
from xml.etree import ElementTree as ET

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import Json, execute_values

from src.chunking.tags import extract_meta_tags, roman_or_integer as _roman_or_integer


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_BOOKS_DIR = PROJECT_ROOT / "books" / "vronsky"
DEFAULT_BATCH_SIZE = 500
SCHOOL_NAME = "Западная классическая астрология"
CANONICAL_AUTHOR = "С. А. Вронский"
MAX_CHUNK_CHARS = 1500

LOGGER = logging.getLogger("parse_and_ingest")

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?…])\s+(?=[А-ЯЁA-Z0-9«\"(])")
VOLUME_IN_TITLE = re.compile(r"\b(?:том|volume|tom)[\s_-]*(\d{1,2})\b", re.IGNORECASE)


@dataclass(frozen=True)
class BookMetadata:
    title: str
    author_names: tuple[str, ...]
    volume_num: int | None
    volume_title: str | None
    description: str | None


@dataclass(frozen=True)
class ChunkData:
    chapter_title: str | None
    section_title: str | None
    hierarchy_path: list[str]
    raw_text: str
    clean_text: str
    meta_tags: dict[str, list[object]]


def _local_name(tag: str) -> str:
    """Return an XML element's local name, ignoring its namespace."""
    return tag.rsplit("}", 1)[-1].split(":")[-1].lower()


def _child(element: ET.Element, name: str) -> ET.Element | None:
    return next((node for node in element if _local_name(node.tag) == name), None)


def _children(element: ET.Element, name: str) -> list[ET.Element]:
    return [node for node in element if _local_name(node.tag) == name]


def _text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return " ".join(" ".join(element.itertext()).split())


def _author_name(author: ET.Element) -> str:
    parts = [
        _text(_child(author, field))
        for field in ("first-name", "middle-name", "last-name")
    ]
    name = " ".join(part for part in parts if part).strip()
    if not name:
        name = _text(_child(author, "nickname"))
    return name


def _parse_metadata(root: ET.Element, source_name: str) -> BookMetadata:
    title_info = next(
        (node for node in root.iter() if _local_name(node.tag) == "title-info"),
        None,
    )
    if title_info is None:
        raise ValueError("FB2 file has no <title-info> metadata")

    book_title = _text(_child(title_info, "book-title"))
    if not book_title:
        raise ValueError("FB2 <title-info> has no <book-title>")

    sequence = next(
        (node for node in title_info.iter() if _local_name(node.tag) == "sequence"),
        None,
    )
    sequence_name = (sequence.attrib.get("name", "").strip() if sequence is not None else "")
    raw_volume = sequence.attrib.get("number", "").strip() if sequence is not None else ""
    volume_num = _roman_or_integer(raw_volume) if raw_volume else None
    if volume_num is None:
        title_volume_match = VOLUME_IN_TITLE.search(book_title)
        if title_volume_match:
            volume_num = int(title_volume_match.group(1))
    if volume_num is None:
        source_volume_match = VOLUME_IN_TITLE.search(Path(source_name.split("!")[0]).stem)
        if source_volume_match:
            volume_num = int(source_volume_match.group(1))

    authors = tuple(
        name
        for author in _children(title_info, "author")
        if (name := _author_name(author))
    )
    description = _text(_child(title_info, "annotation")) or None

    return BookMetadata(
        title=sequence_name or book_title,
        author_names=authors,
        volume_num=volume_num,
        volume_title=book_title,
        description=description,
    )


def _split_oversized_paragraph(paragraph: str, limit: int) -> list[str]:
    """Split a long paragraph at sentence boundaries, keeping sentences intact."""
    sentences = [sentence.strip() for sentence in SENTENCE_BOUNDARY.split(paragraph) if sentence.strip()]
    if len(sentences) <= 1:
        return [paragraph]

    result: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > limit:
            result.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        result.append(current)
    return result


def _group_paragraphs(paragraphs: list[str], limit: int) -> list[str]:
    """Group paragraphs near the target size without splitting sentences."""
    pieces: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) > limit:
            pieces.extend(_split_oversized_paragraph(paragraph, limit))
        else:
            pieces.append(paragraph)

    chunks: list[str] = []
    current: list[str] = []
    current_size = 0
    for paragraph in pieces:
        added_size = len(paragraph) + (2 if current else 0)
        if current and current_size + added_size > limit:
            chunks.append("\n\n".join(current))
            current = [paragraph]
            current_size = len(paragraph)
        else:
            current.append(paragraph)
            current_size += added_size
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _section_title(section: ET.Element) -> str:
    title = _child(section, "title")
    return _text(title)


def _section_paragraphs(section: ET.Element) -> list[str]:
    """Get this section's own text blocks, excluding nested sections."""
    paragraphs: list[str] = []
    for child in section:
        name = _local_name(child.tag)
        if name == "section":
            continue
        if name == "title":
            continue
        value = _text(child)
        if value:
            paragraphs.append(value)
    return paragraphs


def _book_chunks(root: ET.Element, metadata: BookMetadata) -> list[ChunkData]:
    chunks: list[ChunkData] = []
    volume_label = (
        f"Том {metadata.volume_num}"
        if metadata.volume_num is not None
        else metadata.volume_title
    )
    base_path = [volume_label] if volume_label else []

    def visit(section: ET.Element, parent_path: list[str]) -> None:
        title = _section_title(section)
        path = parent_path + ([title] if title else [])
        paragraphs = _section_paragraphs(section)
        if paragraphs:
            heading_context = " > ".join(path)
            heading_prefix = f"{heading_context}\n\n" if heading_context else ""
            
            for body in _group_paragraphs(paragraphs, MAX_CHUNK_CHARS):
                raw_chunk = "\n\n".join(([title] if title else []) + [body])
                clean_body = " ".join(body.split())
                
                # Формируем Markdown-заголовок для сохранения структуры
                heading_formatted = f"## {' > '.join(path)}" if path else ""
                clean_text = f"{heading_formatted}\n\n{clean_body}".strip()
                
                chunks.append(
                    ChunkData(
                        chapter_title=path[1] if len(path) > 1 else (path[0] if path else None),
                        section_title=path[-1] if path else None,
                        hierarchy_path=path,
                        raw_text=raw_chunk,
                        clean_text=clean_text,
                        meta_tags=extract_meta_tags(title, clean_text),
                    )
                )
        for nested in _children(section, "section"):
            visit(nested, path)

    bodies = [node for node in root.iter() if _local_name(node.tag) == "body"]
    found_sections = False
    for body in bodies:
        for section in _children(body, "section"):
            found_sections = True
            visit(section, base_path)

    if not found_sections:
        raise ValueError("FB2 file contains no <body>/<section> structure")
    return chunks


def parse_fb2(data: bytes, source_name: str) -> tuple[BookMetadata, list[ChunkData]]:
    """Parse a FB2 XML document into metadata and section-aware chunks."""
    root = ET.fromstring(data)
    metadata = _parse_metadata(root, source_name)
    chunks = _book_chunks(root, metadata)
    if not chunks:
        raise ValueError("FB2 file contains no text paragraphs in its sections")
    return metadata, chunks


def _iter_book_files(directory: Path) -> Iterator[tuple[str, bytes]]:
    """Yield standalone FB2 files and FB2 documents stored inside ZIP archives."""
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == ".fb2":
            try:
                yield str(path), path.read_bytes()
            except OSError:
                LOGGER.exception("Could not read FB2 file %s", path)
        elif suffix == ".zip":
            try:
                with zipfile.ZipFile(path) as archive:
                    for member in sorted(archive.namelist()):
                        if member.lower().endswith(".fb2") and not member.endswith("/"):
                            try:
                                yield f"{path}!{member}", archive.read(member)
                            except (OSError, RuntimeError, zipfile.BadZipFile):
                                LOGGER.exception("Could not read %s from archive %s", member, path)
            except (OSError, zipfile.BadZipFile):
                LOGGER.exception("Could not open ZIP archive %s", path)


def _ensure_school(cursor) -> int:
    cursor.execute(
        "INSERT INTO schools (name) VALUES (%s) ON CONFLICT (name) DO NOTHING",
        (SCHOOL_NAME,),
    )
    cursor.execute("SELECT id FROM schools WHERE name = %s", (SCHOOL_NAME,))
    row = cursor.fetchone()
    if row is None:
        raise RuntimeError(f"Could not create or find school {SCHOOL_NAME!r}")
    return row[0]


def _ensure_author(cursor, name: str) -> int:
    cursor.execute("SELECT id FROM authors WHERE name = %s ORDER BY id LIMIT 1", (name,))
    row = cursor.fetchone()
    if row is not None:
        return row[0]
    cursor.execute(
        "INSERT INTO authors (name) VALUES (%s) RETURNING id",
        (name,),
    )
    return cursor.fetchone()[0]


def _ensure_book(cursor, metadata: BookMetadata, school_id: int) -> int:
    cursor.execute(
        """SELECT id FROM books
           WHERE title = %s AND volume_num IS NOT DISTINCT FROM %s
             AND school_id = %s
           ORDER BY id LIMIT 1""",
        (metadata.title, metadata.volume_num, school_id),
    )
    row = cursor.fetchone()
    if row is None:
        cursor.execute(
            """INSERT INTO books (school_id, title, volume_num, volume_title, description)
               VALUES (%s, %s, %s, %s, %s) RETURNING id""",
            (
                school_id,
                metadata.title,
                metadata.volume_num,
                metadata.volume_title,
                metadata.description,
            ),
        )
        return cursor.fetchone()[0]

    book_id = row[0]
    cursor.execute(
        """UPDATE books SET volume_title = %s, description = %s, school_id = %s
           WHERE id = %s""",
        (metadata.volume_title, metadata.description, school_id, book_id),
    )
    return book_id


def _save_book(cursor, metadata: BookMetadata, chunks: list[ChunkData], batch_size: int) -> int:
    school_id = _ensure_school(cursor)
    book_id = _ensure_book(cursor, metadata, school_id)
    author_names = list(dict.fromkeys((CANONICAL_AUTHOR, *metadata.author_names)))
    author_ids = [_ensure_author(cursor, name) for name in author_names]
    for author_id in author_ids:
        cursor.execute(
            """INSERT INTO book_authors (book_id, author_id) VALUES (%s, %s)
               ON CONFLICT (book_id, author_id) DO NOTHING""",
            (book_id, author_id),
        )

    # Re-ingesting the same volume replaces its chunks rather than duplicating them.
    cursor.execute("DELETE FROM chunks WHERE book_id = %s", (book_id,))
    values = []
    for index, chunk in enumerate(chunks):
        chunk_id = uuid5(NAMESPACE_URL, f"astrology-book:{book_id}:chunk:{index}")
        values.append(
            (
                str(chunk_id),
                book_id,
                chunk.chapter_title,
                chunk.section_title,
                chunk.hierarchy_path,
                chunk.raw_text,
                chunk.clean_text,
                Json(chunk.meta_tags, dumps=lambda value: json.dumps(value, ensure_ascii=False)),
            )
        )

    query = """INSERT INTO chunks
        (id, book_id, chapter_title, section_title, hierarchy_path,
         raw_text, clean_text, meta_tags)
        VALUES %s"""
    for start in range(0, len(values), batch_size):
        execute_values(
            cursor,
            query,
            values[start : start + batch_size],
            template="(%s::uuid, %s, %s, %s, %s, %s, %s, %s)",
            page_size=batch_size,
        )
    return len(values)


def _database_url() -> str:
    load_dotenv(PROJECT_ROOT / ".env")
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is missing. Set it in the project .env file or environment."
        )
    if database_url.startswith("postgresql+asyncpg://"):
        database_url = "postgresql://" + database_url.removeprefix("postgresql+asyncpg://")
    return database_url


def ingest_directory(directory: Path, batch_size: int = DEFAULT_BATCH_SIZE) -> tuple[int, int, int]:
    """Parse and transactionally ingest every FB2 book in a directory."""
    if not directory.is_dir():
        raise NotADirectoryError(f"Books directory does not exist: {directory}")
    if batch_size < 1:
        raise ValueError("batch_size must be greater than zero")

    connection = psycopg2.connect(_database_url())
    processed_books = 0
    processed_chunks = 0
    inserted_chunks = 0
    try:
        for source_name, data in _iter_book_files(directory):
            try:
                metadata, chunks = parse_fb2(data, source_name)
                processed_chunks += len(chunks)
                with connection:
                    with connection.cursor() as cursor:
                        saved_count = _save_book(cursor, metadata, chunks, batch_size)
                processed_books += 1
                inserted_chunks += saved_count
                LOGGER.info(
                    "Book %s: parsed %d chunks; successfully wrote %d chunks "
                    "(total processed=%d, total written=%d)",
                    source_name,
                    len(chunks),
                    saved_count,
                    processed_chunks,
                    inserted_chunks,
                )
            except Exception:
                connection.rollback()
                LOGGER.exception("Failed to ingest %s; continuing with next book", source_name)
        return processed_books, processed_chunks, inserted_chunks
    finally:
        connection.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--books-dir",
        type=Path,
        default=DEFAULT_BOOKS_DIR,
        help=f"Directory containing FB2 files or ZIP archives (default: {DEFAULT_BOOKS_DIR})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Chunks per PostgreSQL execute_values batch (default: {DEFAULT_BATCH_SIZE})",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    books, processed, written = ingest_directory(args.books_dir, args.batch_size)
    LOGGER.info(
        "Finished: %d books ingested, %d chunks processed, %d chunks written",
        books,
        processed,
        written,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
