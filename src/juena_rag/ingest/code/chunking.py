"""One file in, a list of :class:`CodeChunk` out.

The splitting matches juena-chatbot's on purpose: a benchmark comparing the two
stores should measure storage, not different chunk boundaries. Each chunk also
records its line range, enclosing symbol, ``chunk_kind`` and ``is_doc``.
"""

from pathlib import Path
from typing import Any

from juena_rag.parsers import Definition, enclosing, outline
from juena_rag.schema.registry import CodeChunk

try:
    from chonkie import CodeChunker
    from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover - depends on the install, not the code
    raise ImportError(
        "chunking needs `chonkie` and `langchain-text-splitters`, which are not "
        "installed. `pip install 'juena-rag[ingest]'`."
    ) from None

#: What a file's extension says it is written in. Anything unlisted is ``"other"``,
#: which still chunks -- just without syntax-aware boundaries.
LANGUAGE_BY_EXTENSION = {
    ".py": "python",
    ".md": "markdown",
    ".rst": "rst",
    ".txt": "text",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hh": "cpp",
    ".hpp": "cpp",
    ".hxx": "cpp",
    ".cl": "c",
    ".f": "fortran",
    ".F": "fortran",
    ".for": "fortran",
    ".f90": "fortran",
    ".F90": "fortran",
    ".inc": "fortran",
    ".tex": "latex",
}

#: Languages chonkie can parse. Everything else goes through the recursive
#: splitter below. Fortran and LaTeX work because `chonkie[code]` already
#: carries their tree-sitter grammars.
_CHONKIE_LANGUAGES = {"python", "markdown", "fortran", "latex"}

#: Languages langchain ships syntax-aware separators for, used only when chonkie
#: has no grammar or its parse comes back empty. Fortran has no ``Language``
#: member, so it would split on blank lines -- which is why it goes to chonkie.
_LANGCHAIN_LANGUAGES = {
    "python": Language.PYTHON,
    "markdown": Language.MARKDOWN,
    "rst": Language.RST,
    "c": Language.C,
    "cpp": Language.CPP,
}

#: Languages whose chunks are prose, so they stay ``chunk_kind="text"``.
#:
#: Keyed on the language, not on ``is_doc``: ``is_doc`` says where the file lives
#: (the configured docs tree), ``chunk_kind`` says what the text is. A README
#: outside the docs tree is still prose.
_PROSE_LANGUAGES = {"markdown", "rst", "text", "latex"}


def detect_language(rel_path: str) -> str:
    """Tell a file's language from its extension.

    Args:
        rel_path: The file's path inside the repository.

    Returns:
        The language name, or ``"other"`` for an unlisted extension.
    """
    return LANGUAGE_BY_EXTENSION.get(Path(rel_path).suffix, "other")


# --------------------------------------------------------------------------- #
# Splitting -- two backends producing the same shape
# --------------------------------------------------------------------------- #


def _chonkie_split(source: str, language: str, chunk_size: int) -> list[dict[str, Any]]:
    """Parser-backed splitting, so a chunk tends to end where a function does.

    Returns an empty list when chonkie cannot parse the file, which puts it
    through the recursive splitter instead of failing the whole ingest.
    """
    try:
        chunker = CodeChunker(
            language=language, tokenizer="character", chunk_size=chunk_size
        )
        pieces = chunker.chunk(source)
    except Exception:  # noqa: BLE001 -- see the docstring; the parser is a black box
        return []

    return [
        {
            "text": piece.text,
            "start_index": piece.start_index,
            "end_index": piece.end_index,
            "token_count": piece.token_count,
        }
        for piece in pieces
    ]


def _recursive_split(
    source: str, language: str, chunk_size: int, chunk_overlap: int
) -> list[dict[str, Any]]:
    """Split on the largest separator that fits, then the next largest, and so on.

    ``add_start_index`` is what makes the offsets trustworthy: searching for the
    chunk text afterwards fails whenever the splitter stripped whitespace.
    """
    settings: dict[str, Any] = {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "length_function": len,
        "add_start_index": True,
    }

    known = _LANGCHAIN_LANGUAGES.get(language)
    splitter = (
        RecursiveCharacterTextSplitter.from_language(known, **settings)
        if known is not None
        else RecursiveCharacterTextSplitter(**settings)
    )

    return [
        {
            "text": document.page_content,
            "start_index": document.metadata.get("start_index", 0),
            "end_index": document.metadata.get("start_index", 0)
            + len(document.page_content),
            "token_count": len(document.page_content),
        }
        for document in splitter.create_documents([source])
    ]


def _cap_oversized(
    pieces: list[dict[str, Any]], language: str, chunk_size: int, chunk_overlap: int
) -> list[dict[str, Any]]:
    """Re-split anything chonkie returned that is over the size limit anyway.

    A parser has to keep a syntactic unit whole, so one 8,000-character function
    comes back as one chunk however small ``chunk_size`` is.
    """
    capped: list[dict[str, Any]] = []

    for piece in pieces:
        text = piece["text"]
        if (
            len(text) <= chunk_size
            and piece.get("token_count", len(text)) <= chunk_size
        ):
            capped.append(piece)
            continue

        # The children's offsets are relative to the parent, so they need shifting
        # back into the file's own coordinates.
        parent = piece.get("start_index", 0)
        for child in _recursive_split(text, language, chunk_size, chunk_overlap):
            capped.append(
                {
                    **child,
                    "start_index": parent + child["start_index"],
                    "end_index": parent + child["end_index"],
                }
            )

    return capped


# --------------------------------------------------------------------------- #
# The two things derived from the file's own structure
# --------------------------------------------------------------------------- #


def _dotted(definitions: list[Definition], first: int, last: int) -> str | None:
    """The chain enclosing the whole chunk as one name -- ``"FormFactor.sphere"``."""
    found = enclosing(definitions, first, last)
    return None if found is None else found.dotted


def _headings(definitions: list[Definition], first: int, last: int) -> list[str]:
    """The chain enclosing the whole chunk as a list -- ``["Fitting", "Least squares"]``."""
    found = enclosing(definitions, first, last)
    return [] if found is None else list(found.path)


def line_of(source: str, offset: int) -> int:
    """Find the line a character offset falls on.

    Args:
        source: The file's text.
        offset: A character position in ``source``.

    Returns:
        The line number, counting from 1.
    """
    return source.count("\n", 0, offset) + 1


# --------------------------------------------------------------------------- #
# The public call
# --------------------------------------------------------------------------- #


def chunk_file(
    source: str,
    rel_path: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
    is_doc: bool = False,
) -> list[CodeChunk]:
    """Split one file into chunks ready to hand to a ``ChunkBatch``.

    ``chunk_size`` is in characters, not tokens; the embedding model's limit is in
    tokens, which ``ModelTokenCounter`` measures. Both sizes are required and come
    from the caller's ``repositories.yaml``.

    Args:
        source: The file's text.
        rel_path: The file's path inside the repository. It decides the language
            and becomes each chunk's ``source_uri``.
        chunk_size: The largest chunk, in characters.
        chunk_overlap: How many characters consecutive chunks may share. Clamped
            to below ``chunk_size``.
        is_doc: Whether the file is documentation rather than source.

    Returns:
        The file's chunks in order, without blank ones.
    """
    language = detect_language(rel_path)
    # The splitter refuses an overlap at or above the chunk size (1500/200 is
    # fine, 100/200 is not). Clamped once here so both paths below agree.
    chunk_overlap = min(chunk_overlap, max(chunk_size - 1, 0))

    pieces: list[dict[str, Any]] = []
    if language in _CHONKIE_LANGUAGES:
        pieces = _chonkie_split(source, language, chunk_size)

    if pieces:
        pieces = _cap_oversized(pieces, language, chunk_size, chunk_overlap)
    else:
        pieces = _recursive_split(source, language, chunk_size, chunk_overlap)

    definitions = outline(language, source)
    kind = "text" if language in _PROSE_LANGUAGES else "code_block"
    # A LaTeX heading is not a definition, so it belongs in the field meant for a
    # document's structure. Everything else names the code it sits inside.
    as_heading = language == "latex"

    chunks: list[CodeChunk] = []
    for piece in pieces:
        # A splitter can hand back a run of blank lines, which `Chunk` rejects:
        # its vector would match everything weakly and nothing well.
        if not piece["text"].strip():
            continue

        start_index = piece.get("start_index", 0)
        end_index = piece.get("end_index", start_index + len(piece["text"]))
        start_line = line_of(source, start_index)
        end_line = line_of(source, max(end_index - 1, start_index))

        chunks.append(
            CodeChunk(
                text=piece["text"],
                source_uri=rel_path,
                language=language,
                chunk_kind=kind,
                is_doc=is_doc,
                # `end_index` points one past the last character, so a chunk ending
                # in a newline would otherwise claim the line after itself.
                start_line=start_line,
                end_line=end_line,
                start_index=start_index,
                end_index=end_index,
                # Named for what encloses the chunk *entirely*, not for what its
                # first line happens to sit in -- see `enclosing`.
                symbol=None
                if as_heading
                else _dotted(definitions, start_line, end_line),
                heading_path=(
                    _headings(definitions, start_line, end_line) if as_heading else []
                ),
            )
        )

    return chunks
