"""What a file turns into: chunk kinds, line ranges, and the enclosing symbol.

Nothing here embeds anything. The chunkers are deterministic, so these assert what
lands on a `CodeChunk` rather than where exactly a boundary fell -- boundary
choices belong to chonkie and langchain, which are not this pipeline's to test.
"""

import pytest

pytest.importorskip("chonkie", reason="needs the `[ingest]` extra")

from juena_rag.ingest.code.chunking import (  # imported below the skip, deliberately
    chunk_file,
    detect_language,
    line_of,
)

#: The sizes a test uses when its point is not the size. `chunk_file` has no
#: default -- the caller's config always says -- so the tests say too.
SIZES = {"chunk_size": 1500, "chunk_overlap": 200}


def split(source: str, rel_path: str, **settings):
    """`chunk_file` with SIZES, so a test names only the setting it is about."""
    return chunk_file(source, rel_path, **(SIZES | settings))

SOURCE = '''"""A module."""


def scattering_length(element):
    """Look one up."""
    return TABLE[element]


class FormFactor:
    """A shape."""

    def sphere(self, q, radius):
        return 3 * (sin(q * radius) - q * radius * cos(q * radius))
'''


# --------------------------------------------------------------------------- #
# Language and kind
# --------------------------------------------------------------------------- #


def test_the_extension_decides_the_language():
    assert detect_language("src/jscatter/formel.py") == "python"
    assert detect_language("doc/source/sas.rst") == "rst"
    assert detect_language("src/dtr/fit.f90") == "fortran"
    # Absent from the table until datreat's manual turned up chunked as source.
    assert detect_language("doc/NewManualBook/ManualDtr.tex") == "latex"
    assert detect_language("Makefile") == "other"


def test_source_chunks_are_code_and_prose_chunks_are_text():
    """`chunk_kind` asks what the text *is*, not where the file lives.

    The partial index `chunks_kind_idx` covers everything that is not `"text"`, so
    marking prose as code would put the whole documentation tree in it for nothing.
    """
    code = split(SOURCE, "src/jscatter/formel.py")
    prose = split("Some heading\n============\n\nWords.\n", "doc/sas.rst")

    assert {chunk.chunk_kind for chunk in code} == {"code_block"}
    assert {chunk.chunk_kind for chunk in prose} == {"text"}


def test_is_doc_is_carried_from_the_caller_not_guessed():
    chunks = split(SOURCE, "src/jscatter/formel.py", is_doc=True)
    assert all(chunk.is_doc for chunk in chunks)


# --------------------------------------------------------------------------- #
# Symbols
# --------------------------------------------------------------------------- #


def test_a_chunk_is_named_for_the_definition_it_starts_in():
    """Small `chunk_size` on purpose, so the file splits and a chunk starts inside.

    A whole short file is one chunk starting at line 1, which is outside every
    definition -- see the test below, which is the more common case.
    """
    body = "\n".join(f"    value_{n} = {n}" for n in range(200))
    chunks = split(f"def huge():\n{body}\n", "formel.py", chunk_size=400)

    named = {chunk.symbol for chunk in chunks if chunk.symbol}
    assert named == {"huge"}


def test_a_chunk_starting_above_the_first_definition_has_no_symbol():
    """Worth knowing before reading the field: most chunks will have none.

    At `chunk_size=1500` a short module is one chunk beginning at its imports, and
    `symbol` earns its place only on files with definitions longer than a chunk.
    """
    chunks = split(SOURCE, "formel.py")
    assert len(chunks) == 1
    assert chunks[0].symbol is None


def test_a_language_with_no_parser_yields_no_symbols():
    """reStructuredText has no grammar here, so text that looks like code is text."""
    chunks = split("Heading\n=======\n\ndef looks_like_python():\n", "a.rst")
    assert all(chunk.symbol is None for chunk in chunks)


def test_a_fortran_chunk_is_named_for_its_subroutine():
    """The whole of datreat depends on this: 98% Fortran, and `ast` cannot read it."""
    source = (
        "      subroutine formfactor(q, r, f)\n"
        + "".join(f"      x{n} = {n}\n" for n in range(200))
        + "      end subroutine formfactor\n"
    )
    chunks = split(source, "src/ff.f90", chunk_size=400)
    assert {chunk.symbol for chunk in chunks} == {"formfactor"}


def test_latex_is_prose_carrying_a_heading_path_rather_than_a_symbol():
    """Two separate mistakes it used to make: labelled `code_block`, and unstructured.

    `chunk_kind` decides whether `--code-only` returns the manual, and a heading is
    a document's structure -- `symbol` is for a definition, which a section is not.
    """
    source = (
        "\\chapter{Data Treatment}\n"
        + "\\section{Fitting}\n"
        + "".join(f"Line {n} of the manual.\n" for n in range(200))
    )
    chunks = split(source, "doc/manual.tex")
    assert {chunk.chunk_kind for chunk in chunks} == {"text"}
    assert all(chunk.symbol is None for chunk in chunks)
    assert any(chunk.heading_path == ["Data Treatment", "Fitting"] for chunk in chunks)


# --------------------------------------------------------------------------- #
# Offsets and lines
# --------------------------------------------------------------------------- #


def test_line_numbers_start_at_one():
    assert line_of("a\nb\nc", 0) == 1
    assert line_of("a\nb\nc", 2) == 2


def test_a_chunk_ending_in_a_newline_does_not_claim_the_next_line():
    """`end_index` points one past the last character, so this is off by one easily."""
    chunks = split("alpha\nbeta\n", "a.txt", chunk_size=100, chunk_overlap=0)
    assert len(chunks) == 1
    assert (chunks[0].start_line, chunks[0].end_line) == (1, 2)


def test_an_overlap_wider_than_the_chunk_is_clamped_rather_than_raising():
    """1500/200 is a sensible config and 100/200 is not, but only one is written down.

    The splitter refuses the second outright, which would turn a `--limit` smoke
    test with a small chunk size into a crash.
    """
    chunks = split(
        "alpha\nbeta\ngamma\n", "a.txt", chunk_size=8, chunk_overlap=200
    )
    assert chunks


def test_the_offsets_point_back_at_the_text_they_came_from():
    """The whole reason `add_start_index` is on: searching for the text can't work."""
    chunks = split(SOURCE, "formel.py")
    for chunk in chunks:
        assert SOURCE[chunk.start_index : chunk.end_index] == chunk.text


def test_every_chunk_covers_lines_in_order():
    for chunk in split(SOURCE, "formel.py"):
        assert chunk.start_line <= chunk.end_line


# --------------------------------------------------------------------------- #
# What must never reach the store
# --------------------------------------------------------------------------- #


def test_a_whitespace_only_chunk_is_dropped_rather_than_rejected_later():
    """`Chunk` refuses blank text, so a run of empty lines would fail the batch.

    The chatbot never validated, so it never noticed; here it would stop an ingest
    of two thousand chunks over one of them.
    """
    chunks = split("code = 1\n" + "\n" * 400 + "more = 2\n", "a.py", chunk_size=50)
    assert chunks
    assert all(chunk.text.strip() for chunk in chunks)


def test_a_file_with_nothing_in_it_yields_nothing():
    assert split("\n\n   \n", "empty.py") == []


def test_an_oversized_unit_is_split_rather_than_passed_through():
    """A parser keeps a function whole however small `chunk_size` is."""
    body = "\n".join(f"    value_{n} = {n}" for n in range(400))
    chunks = split(f"def huge():\n{body}\n", "a.py", chunk_size=500)
    assert len(chunks) > 1
    assert all(len(chunk.text) <= 500 for chunk in chunks)
