"""The CSV listing which exports to ingest, and what only a person knows of each.

One row is one HybridChunker export, which becomes one source::

    raw_chunks,docling_json,title,isbn,ref_chapter,editors,year,publisher
    data/01_Brueckel/01_Brueckel_raw_chunks.json,data/01_Brueckel/01_Brueckel.json,Lecture Notes …,000000000-0,01,,2022,Forschungszentrum Jülich

``raw_chunks`` and ``docling_json`` are required in every row; a relative path
is read from the CSV's own folder. Every other column is optional, and an
empty cell means "not set". ``editors`` holds several names separated by
``;``. The CSV belongs to whoever runs the ingest, like ``repositories.yaml``
for code; juena-rag only reads it.
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: The ``BookChunk`` fields only a person can supply -- neither file has them.
#: Everything else on a chunk comes from the two files.
BOOK_INPUTS = ("isbn", "ref_chapter", "editors", "year", "publisher")

#: Every column the CSV may have. Any other is refused, so a misspelt header
#: is an error rather than a field silently left unset on every chunk.
COLUMNS = ("raw_chunks", "docling_json", "title", *BOOK_INPUTS)

_REQUIRED = ("raw_chunks", "docling_json")


@dataclass
class Export:
    """One row: an export's two files, and what a person adds to them.

    Attributes:
        raw_chunks: The ``*_raw_chunks.json``.
        docling_json: The docling JSON it was chunked from.
        title: The book's title.
        book: Whichever of :data:`BOOK_INPUTS` the row sets.
    """

    raw_chunks: Path
    docling_json: Path
    title: str | None = None
    book: dict[str, Any] = field(default_factory=dict)


def load_manifest(path: Path) -> list[Export]:
    """Read the CSV, one :class:`Export` per row.

    Args:
        path: The CSV file.

    Returns:
        The exports, in the CSV's order.

    Raises:
        ValueError: If a column is unknown or missing, or a cell is invalid;
            the message names the line.
    """
    # utf-8-sig: Excel's "CSV UTF-8" starts the file with a byte-order mark,
    # which would otherwise become part of the first column's name.
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        header = reader.fieldnames or []
        unknown = sorted(set(header) - set(COLUMNS))
        if unknown:
            raise ValueError(
                f"{path.name}: unknown column(s) {unknown}; the columns are "
                f"{list(COLUMNS)}"
            )
        missing = [column for column in _REQUIRED if column not in header]
        if missing:
            raise ValueError(f"{path.name}: no {missing} column")
        return [_export(row, path.parent, reader.line_num) for row in reader]


def _export(row: dict[str | None, Any], folder: Path, line: int) -> Export:
    """Turn one CSV row into an :class:`Export`."""
    if None in row:
        raise ValueError(f"line {line}: more cells than the header has columns")
    cells = {name: (value or "").strip() for name, value in row.items()}

    def path(column: str) -> Path:
        if not cells[column]:
            raise ValueError(f"line {line}: `{column}` is empty")
        value = Path(cells[column])
        return value if value.is_absolute() else folder / value

    book: dict[str, Any] = {}
    for name in BOOK_INPUTS:
        value = cells.get(name)
        if not value:
            continue
        if name == "editors":
            book[name] = [
                editor.strip() for editor in value.split(";") if editor.strip()
            ]
        elif name == "year":
            if not value.isdigit():
                raise ValueError(f"line {line}: `year` must be a number, not {value!r}")
            book[name] = int(value)
        else:
            book[name] = value

    return Export(
        raw_chunks=path("raw_chunks"),
        docling_json=path("docling_json"),
        title=cells.get("title") or None,
        book=book,
    )
