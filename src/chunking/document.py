"""Load FB2 and plain-text books into a section tree used by chunking strategies."""

from __future__ import annotations

import logging
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from uuid import NAMESPACE_URL, uuid5
from xml.etree import ElementTree as ET

from .tags import DEFAULT_SCHOOL, detect_language, infer_school, normalize_language


LOGGER = logging.getLogger(__name__)
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
VOLUME_IN_TITLE = re.compile(r"\b(?:том|volume|tom)[\s_-]*(\d{1,2})\b", re.IGNORECASE)
ROMAN_VOLUMES = {
    "I": 1,
    "II": 2,
    "III": 3,
    "IV": 4,
    "V": 5,
    "VI": 6,
    "VII": 7,
    "VIII": 8,
    "IX": 9,
    "X": 10,
    "XI": 11,
    "XII": 12,
}


@dataclass(frozen=True)
class Section:
    """A single FB2/Markdown section with its own paragraphs and child sections."""

    title: str | None
    paragraphs: tuple[str, ...]
    children: tuple["Section", ...] = ()

    @property
    def text(self) -> str:
        return "\n\n".join(self.paragraphs)


@dataclass(frozen=True)
class BookDocument:
    """A parsed book ready for strategy-specific chunking."""

    source: str
    book_id: str
    title: str
    author: str | None
    series: str | None
    volume: str | None
    sections: tuple[Section, ...]
    language: str = "ru"
    school: str = DEFAULT_SCHOOL

    @property
    def display_title(self) -> str:
        if self.volume:
            return f"{self.title} ({self.volume})"
        return self.title


@dataclass(frozen=True)
class TextBlock:
    """A section fragment with its own text, in reading order."""

    path: tuple[str, ...]
    title: str | None
    paragraphs: tuple[str, ...]

    @property
    def text(self) -> str:
        body = "\n\n".join(self.paragraphs)
        return f"{self.title}\n\n{body}" if self.title else body


@dataclass(frozen=True)
class BlockSpan:
    """Character span of a block inside a flattened document."""

    start: int
    end: int
    path: tuple[str, ...]
    title: str | None


@dataclass
class LoadReport:
    """Parsed documents plus per-file errors that did not stop the run."""

    documents: list[BookDocument]
    errors: list[tuple[str, str]]


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].split(":")[-1].lower()


def _child(element: ET.Element, name: str) -> ET.Element | None:
    return next((node for node in element if _local_name(node.tag) == name), None)


def _children(element: ET.Element, name: str) -> list[ET.Element]:
    return [node for node in element if _local_name(node.tag) == name]


def _text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return " ".join(" ".join(element.itertext()).split())


def _block_text(element: ET.Element) -> str:
    """Flatten a text block, keeping poem/citation lines as separate paragraphs."""
    pieces: list[str] = []
    for node in element.iter():
        if _local_name(node.tag) in {"p", "v", "subtitle", "text-author"}:
            value = _text(node)
            if value:
                pieces.append(value)
    if pieces:
        return "\n".join(pieces)
    return _text(element)


def _author_name(author: ET.Element) -> str:
    parts = [_text(_child(author, field)) for field in ("first-name", "middle-name", "last-name")]
    name = " ".join(part for part in parts if part).strip()
    return name or _text(_child(author, "nickname"))


def _volume_number(value: str) -> int | None:
    value = value.strip().upper()
    if value.isdigit():
        number = int(value)
        return number if 1 <= number <= 99 else None
    return ROMAN_VOLUMES.get(value)


def _metadata(root: ET.Element, source: str) -> tuple[str, str | None, str | None, str | None, str]:
    title_info = next((node for node in root.iter() if _local_name(node.tag) == "title-info"), None)
    if title_info is None:
        raise ValueError("FB2 file has no <title-info> metadata")

    book_title = _text(_child(title_info, "book-title"))
    if not book_title:
        raise ValueError("FB2 <title-info> has no <book-title>")

    sequence = next((node for node in title_info.iter() if _local_name(node.tag) == "sequence"), None)
    series = sequence.attrib.get("name", "").strip() if sequence is not None else None
    raw_volume = sequence.attrib.get("number", "").strip() if sequence is not None else ""
    volume_num = _volume_number(raw_volume) if raw_volume else None
    if volume_num is None:
        match = VOLUME_IN_TITLE.search(book_title) or VOLUME_IN_TITLE.search(Path(source.split("!")[0]).stem)
        if match:
            volume_num = int(match.group(1))

    authors = [name for author in _children(title_info, "author") if (name := _author_name(author))]
    author = ", ".join(authors) or None
    volume = f"Том {volume_num}" if volume_num is not None else None

    annotation = _text(_child(title_info, "annotation"))
    raw_language = _text(_child(title_info, "lang"))
    language = normalize_language(raw_language) or detect_language(f"{book_title}\n{annotation}")
    return book_title, author, series or None, volume, language


def _parse_section(element: ET.Element) -> Section:
    title_element = _child(element, "title")
    title = _text(title_element) or None
    paragraphs: list[str] = []
    for child in element:
        name = _local_name(child.tag)
        if name in {"section", "title"}:
            continue
        value = _block_text(child)
        if value:
            paragraphs.append(value)
    children = tuple(_parse_section(node) for node in _children(element, "section"))
    return Section(title=title, paragraphs=tuple(paragraphs), children=children)


def parse_fb2(data: bytes, source: str, *, school: str = DEFAULT_SCHOOL) -> BookDocument:
    """Parse an FB2 byte stream into metadata and a section tree."""
    root = ET.fromstring(data)
    title, author, series, volume, language = _metadata(root, source)

    sections: list[Section] = []
    for body in _children(root, "body"):
        body_sections = _children(body, "section")
        if body_sections:
            sections.extend(_parse_section(section) for section in body_sections)
            continue
        paragraphs = [
            value
            for child in body
            if _local_name(child.tag) not in {"title", "section"}
            and (value := _block_text(child))
        ]
        if paragraphs:
            sections.append(Section(title=_text(_child(body, "title")) or None, paragraphs=tuple(paragraphs)))

    if not sections:
        raise ValueError("FB2 file contains no text sections")

    identity = f"astrology-book:{author or ''}:{title}:{volume or ''}"
    return BookDocument(
        source=source,
        book_id=str(uuid5(NAMESPACE_URL, identity)),
        title=title,
        author=author,
        series=series,
        volume=volume,
        sections=tuple(sections),
        language=language,
        school=infer_school(author, default=school),
    )


def _clean_text(value: str) -> str:
    return " ".join(value.split())


def _read_text(path: Path) -> str:
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "cp1251"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


class _SectionBuilder:
    def __init__(self, title: str | None) -> None:
        self.title = title
        self.paragraphs: list[str] = []
        self.children: list[_SectionBuilder] = []

    def freeze(self) -> Section:
        return Section(
            title=self.title,
            paragraphs=tuple(self.paragraphs),
            children=tuple(child.freeze() for child in self.children),
        )


def parse_text(text: str, source: str, *, school: str = DEFAULT_SCHOOL) -> BookDocument:
    """Parse Markdown headings or plain text into a section tree."""
    source_path = Path(source)
    stem = source_path.stem or source
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = [block.strip() for block in re.split(r"\n\s*\n", normalized) if block.strip()]
    root_title = stem

    roots: list[_SectionBuilder] = []
    stack: list[tuple[int, _SectionBuilder]] = []
    current: _SectionBuilder | None = None
    saw_heading = False

    for block in blocks:
        lines = block.split("\n")
        heading_match = HEADING.match(lines[0].strip())
        if heading_match and len(lines) == 1:
            saw_heading = True
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            builder = _SectionBuilder(title)
            while stack and stack[-1][0] >= level:
                stack.pop()
            if stack:
                stack[-1][1].children.append(builder)
            else:
                roots.append(builder)
            stack.append((level, builder))
            current = builder
            continue

        paragraph = _clean_text(" ".join(lines))
        if not paragraph:
            continue
        if current is None:
            current = _SectionBuilder(None)
            roots.append(current)
        current.paragraphs.append(paragraph)

    if not saw_heading and len(roots) == 1 and roots[0].title is None:
        roots[0].title = root_title
    elif not roots:
        roots.append(_SectionBuilder(root_title))

    author = None
    identity = f"astrology-text:{source_path.name}:{root_title}"
    return BookDocument(
        source=source,
        book_id=str(uuid5(NAMESPACE_URL, identity)),
        title=root_title,
        author=author,
        series=None,
        volume=None,
        sections=tuple(builder.freeze() for builder in roots),
        language=detect_language(text),
        school=infer_school(author, default=school),
    )


def iter_sections(document: BookDocument) -> Iterator[tuple[tuple[str, ...], Section]]:
    """Yield sections depth-first in reading order with their heading path."""

    def walk(section: Section, parent_path: tuple[str, ...]) -> Iterator[tuple[tuple[str, ...], Section]]:
        path = parent_path + ((section.title,) if section.title else ())
        yield path, section
        for child in section.children:
            yield from walk(child, path)

    for root in document.sections:
        yield from walk(root, (document.volume,) if document.volume else ())


def iter_blocks(document: BookDocument) -> Iterator[TextBlock]:
    """Yield sections that have their own text, in reading order."""
    for path, section in iter_sections(document):
        if section.paragraphs:
            yield TextBlock(path=path, title=section.title, paragraphs=section.paragraphs)


def flatten_document(document: BookDocument) -> tuple[str, list[BlockSpan]]:
    """Join a document into one string and remember where each section starts."""
    pieces: list[str] = []
    spans: list[BlockSpan] = []
    position = 0
    for path, section in iter_sections(document):
        if not section.title and not section.paragraphs:
            continue
        body = "\n\n".join(section.paragraphs)
        block = f"{section.title}\n\n{body}" if section.title and body else (section.title or body)
        if not block:
            continue
        if pieces:
            pieces.append("\n\n")
            position += 2
        start = position
        pieces.append(block)
        position += len(block)
        spans.append(BlockSpan(start=start, end=position, path=path, title=section.title))
    return "".join(pieces), spans


def path_at(spans: list[BlockSpan], position: int) -> tuple[str, ...]:
    """Return the section path that contains a character position."""
    if not spans:
        return ()
    for span in spans:
        if position < span.start:
            return span.path
        if span.start <= position < span.end:
            return span.path
    return spans[-1].path


def load_file(path: Path, source: str | None = None, *, school: str = DEFAULT_SCHOOL) -> list[BookDocument]:
    """Load one FB2, ZIP or text file; ZIP files may contain several FB2 books."""
    label = source or path.name
    suffix = path.suffix.lower()
    if suffix == ".fb2":
        return [parse_fb2(path.read_bytes(), label, school=school)]
    if suffix == ".zip":
        documents: list[BookDocument] = []
        with zipfile.ZipFile(path) as archive:
            for member in sorted(archive.namelist()):
                if member.lower().endswith(".fb2") and not member.endswith("/"):
                    documents.append(parse_fb2(archive.read(member), f"{label}!{member}", school=school))
        return documents
    if suffix in {".txt", ".md", ".markdown"}:
        return [parse_text(_read_text(path), label, school=school)]
    raise ValueError(f"Unsupported book format: {path.suffix or path.name}")


def load_corpus(directory: Path, *, school: str = DEFAULT_SCHOOL) -> LoadReport:
    """Load every supported book under a directory, collecting recoverable errors."""
    if not directory.is_dir():
        raise NotADirectoryError(f"Books directory does not exist: {directory}")

    documents: list[BookDocument] = []
    errors: list[tuple[str, str]] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or "__MACOSX" in path.parts:
            continue
        if path.suffix.lower() not in {".fb2", ".zip", ".txt", ".md", ".markdown"}:
            continue
        relative = path.relative_to(directory).as_posix()
        try:
            documents.extend(load_file(path, relative, school=school))
        except Exception as exc:  # noqa: BLE001 - one broken book must not stop the corpus
            LOGGER.exception("Could not load %s", relative)
            errors.append((relative, f"{type(exc).__name__}: {exc}"))
    return LoadReport(documents=documents, errors=errors)
