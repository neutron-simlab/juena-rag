"""The CSV listing exports to ingest -- `juena_rag.ingest.docling.manifest`.

No database: reading a CSV touches only the file, and the one pipeline test
below stops at a missing file, before anything is embedded.
"""

import pytest

from juena_rag.ingest.docling.manifest import load_manifest
from juena_rag.ingest.docling.pipeline import ingest_docling_manifest

HEADER = "raw_chunks,docling_json,title,isbn,ref_chapter,editors,year,publisher\n"


def a_csv(tmp_path, *rows, header=HEADER):
    path = tmp_path / "books.csv"
    path.write_text(header + "".join(rows), encoding="utf-8")
    return path


def test_a_row_becomes_its_two_files_the_title_and_the_book_fields(tmp_path):
    path = a_csv(
        tmp_path,
        "01/01_raw_chunks.json,01/01.json,Lecture Notes,000000000-0,01,,2022,"
        "Forschungszentrum Jülich\n",
    )
    (export,) = load_manifest(path)

    assert export.raw_chunks == tmp_path / "01" / "01_raw_chunks.json"
    assert export.docling_json == tmp_path / "01" / "01.json"
    assert export.title == "Lecture Notes"
    assert export.book == {
        "isbn": "000000000-0",
        "ref_chapter": "01",
        "year": 2022,
        "publisher": "Forschungszentrum Jülich",
    }


def test_an_empty_cell_leaves_its_field_unset(tmp_path):
    (export,) = load_manifest(a_csv(tmp_path, "a.json,b.json,,,,,,\n"))
    assert export.title is None
    assert export.book == {}


def test_editors_are_separated_by_semicolons(tmp_path):
    (export,) = load_manifest(
        a_csv(tmp_path, "a.json,b.json,,,,T. Brückel; R. Zorn,,\n")
    )
    assert export.book["editors"] == ["T. Brückel", "R. Zorn"]


def test_an_absolute_path_is_kept_as_it_is(tmp_path):
    raw_chunks = tmp_path / "elsewhere" / "a_raw_chunks.json"
    (export,) = load_manifest(a_csv(tmp_path, f"{raw_chunks},b.json,,,,,,\n"))
    assert export.raw_chunks == raw_chunks


def test_a_misspelt_column_is_refused(tmp_path):
    header = HEADER.replace("publisher", "publsher")
    with pytest.raises(ValueError, match="publsher"):
        load_manifest(a_csv(tmp_path, header=header))


def test_a_csv_without_the_docling_json_column_is_refused(tmp_path):
    with pytest.raises(ValueError, match="docling_json"):
        load_manifest(a_csv(tmp_path, header="raw_chunks,title\n"))


def test_a_year_that_is_not_a_number_names_its_line(tmp_path):
    path = a_csv(tmp_path, "a.json,b.json,,,,,2022,\n", "a.json,b.json,,,,,soon,\n")
    with pytest.raises(ValueError, match="line 3"):
        load_manifest(path)


def test_a_row_with_a_missing_file_stops_the_run_before_anything_is_stored(
    tmp_path, monkeypatch, capsys
):
    """A database is named but never reached: every row is built first."""
    monkeypatch.setenv("JUENA_RAG_DSN", "postgresql://never-reached")
    path = a_csv(tmp_path, "missing_raw_chunks.json,missing.json,,,,,,\n")

    assert ingest_docling_manifest(path) == 2
    assert "missing_raw_chunks.json" in capsys.readouterr().err
