"""What a source file declares, and on which lines.

One function per language, all returning :class:`Definition`, so a chunker can
label a chunk with whatever encloses it without knowing the language it is
reading. Splitting a file into chunks is still the caller's business.
"""

from juena_rag.parsers import clike, fortran, latex, python
from juena_rag.parsers.definition import Definition, enclosing

#: Which parser handles which language. A plain dict rather than a registry: the
#: wiring is three lines long and reads better where you are already looking.
OUTLINERS = {
    "python": python.outline,
    "fortran": fortran.outline,
    "latex": latex.outline,
    "c": clike.outline_c,
    "cpp": clike.outline_cpp,
}


def outline(language: str, source: str) -> list[Definition]:
    """Every definition or heading in ``source``, in source order, innermost last.

    An unparsed language returns an empty list rather than raising, so a caller
    can offer any file it likes and take what comes back.
    """
    parser = OUTLINERS.get(language)
    return [] if parser is None else parser(source)


__all__ = [
    "OUTLINERS",
    "Definition",
    "clike",
    "enclosing",
    "fortran",
    "latex",
    "outline",
    "python",
]
