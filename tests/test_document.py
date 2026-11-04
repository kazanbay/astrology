import zipfile
from pathlib import Path

import pytest

from chunking.document import (
    flatten_document,
    iter_blocks,
    iter_sections,
    load_corpus,
    parse_fb2,
    parse_text,
    path_at,
)


FB2 = """<?xml version="1.0" encoding="windows-1251"?>
<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0">
  <description>
    <title-info>
      <author>
        <first-name>Сергей</first-name>
        <middle-name>Алексеевич</middle-name>
        <last-name>Вронский</last-name>
      </author>
      <book-title>Домология</book-title>
      <sequence name="Классическая астрология" number="3"/>
    </title-info>
  </description>
  <body>
    <title><p>Домология</p></title>
    <section>
      <title><p>Глава 1. Дома</p></title>
      <p>Первый дом описывает личность.</p>
      <poem><v>Строка один</v><v>Строка два</v></poem>
      <section>
        <title><p>1.1. Первый дом</p></title>
        <p>Первый дом связан с Сатурном и VII домом.</p>
      </section>
    </section>
    <section>
      <title><p>Глава 2. Планеты</p></title>
      <p>Луна в первом доме усиливает эмоции.</p>
    </section>
  </body>
</FictionBook>
"""


def test_parse_fb2_handles_windows_1251_and_sections():
    document = parse_fb2(FB2.encode("cp1251"), "book.fb2")

    assert document.title == "Домология"
    assert document.author == "Сергей Алексеевич Вронский"
    assert document.series == "Классическая астрология"
    assert document.volume == "Том 3"
    assert document.display_title == "Домология (Том 3)"
    assert len(document.sections) == 2

    paths = [path for path, _section in iter_sections(document)]
    assert paths == [
        ("Том 3", "Глава 1. Дома"),
        ("Том 3", "Глава 1. Дома", "1.1. Первый дом"),
        ("Том 3", "Глава 2. Планеты"),
    ]

    first_block = next(iter_blocks(document))
    assert first_block.path == ("Том 3", "Глава 1. Дома")
    assert "Строка один\nСтрока два" in first_block.text


def test_flatten_document_maps_positions_to_section_paths():
    document = parse_fb2(FB2.encode("cp1251"), "book.fb2")
    text, spans = flatten_document(document)

    assert text.startswith("Глава 1. Дома")
    assert spans[0].path == ("Том 3", "Глава 1. Дома")
    assert path_at(spans, spans[1].start + 1) == ("Том 3", "Глава 1. Дома", "1.1. Первый дом")
    assert text[spans[1].start : spans[1].end].startswith("1.1. Первый дом")


def test_parse_markdown_builds_nested_sections():
    document = parse_text(
        "# Глава 1\n\nТекст главы.\n\n## 1.1. Раздел\n\nТекст раздела.\n\n# Глава 2\n\nЕщё текст.",
        "notes.md",
    )
    paths = [path for path, _section in iter_sections(document)]
    assert paths == [("Глава 1",), ("Глава 1", "1.1. Раздел"), ("Глава 2",)]
    assert iter_blocks(document).__next__().text == "Глава 1\n\nТекст главы."


def test_plain_text_becomes_one_titled_section():
    document = parse_text("Первый абзац.\n\nВторой абзац.", "plain.txt")
    assert document.title == "plain"
    assert [section.title for section in document.sections] == ["plain"]
    assert document.sections[0].paragraphs == ("Первый абзац.", "Второй абзац.")


def test_load_corpus_reads_fb2_inside_zip_and_reports_broken_files(tmp_path: Path):
    archive = tmp_path / "book.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("book.fb2", FB2)
    (tmp_path / "broken.fb2").write_bytes(b"<FictionBook>")
    (tmp_path / "ignore.pdf").write_bytes(b"%PDF")

    report = load_corpus(tmp_path)

    assert len(report.documents) == 1
    assert report.documents[0].source == "book.zip!book.fb2"
    assert [source for source, _error in report.errors] == ["broken.fb2"]


def test_load_file_rejects_unsupported_format(tmp_path: Path):
    path = tmp_path / "book.pdf"
    path.write_bytes(b"%PDF")
    with pytest.raises(ValueError, match="Unsupported book format"):
        from chunking.document import load_file

        load_file(path)
