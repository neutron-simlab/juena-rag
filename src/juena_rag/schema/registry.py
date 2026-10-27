"""How a category gets its name, and seven worked examples of claiming one.

A **category** says what kind of document a chunk came from. Each is claimed by a
:class:`Chunk` subclass adding the fields that kind of document needs::

    @register_chunk_class
    class SeminarChunk(Chunk):
        category: Literal["seminar"] = "seminar"
        speaker: str | None = None

Write that in **your own code** and ``juena-rag`` validates a batch declaring
``category="seminar"`` -- no edit to this package, no release. The seven classes
below are the ones shipped, and the examples to copy.
"""

from typing import Any, Literal, Self

from pydantic import Field, ValidationError, model_validator

from juena_rag.schema.models import Chunk, PageProvenance
from juena_rag.schema.vocabulary import check_range

#: Category name -> the class that claims it.
_REGISTRY: dict[str, type[Chunk]] = {}


def register_chunk_class(cls: type[Chunk]) -> type[Chunk]:
    """Register a chunk class under the ``category`` it pins. Use as a decorator.

    A registry rather than a discriminated union, because a union closes the
    vocabulary: adding a category would mean a release of this package.

    Explicit rather than automatic on subclassing, so registration is visible
    where a reader is already looking -- and so :class:`LayoutChunk` stays out of
    it by simply not being decorated.
    """
    field = cls.model_fields.get("category")
    category = field.default if field is not None else None
    if not isinstance(category, str) or not category:
        raise TypeError(
            f"{cls.__qualname__} pins no category, so there is nothing to register "
            "it under. Add a pinned field, e.g. "
            '`category: Literal["seminar"] = "seminar"`.'
        )

    existing = _REGISTRY.get(category)
    if existing is not None and existing is not cls:
        raise TypeError(
            f"category {category!r} is already registered to "
            f"{existing.__module__}.{existing.__qualname__}; "
            f"{cls.__module__}.{cls.__qualname__} cannot claim it too. Two classes "
            "claiming one category means which one wins depends on import order."
        )

    _REGISTRY[category] = cls
    return cls


def registered_categories() -> dict[str, type[Chunk]]:
    """Every category a chunk class has claimed, newest registration last."""
    return dict(_REGISTRY)


def chunk_class(category: str) -> type[Chunk]:
    """The class for ``category``, or plain :class:`Chunk` if none is registered.

    Falling back is legitimate: a caller with no category-specific fields to
    carry should not have to define a class in order to say so.
    """
    return _REGISTRY.get(category, Chunk)


def parse_chunk(category: str, data: Any) -> Chunk:
    """Validate ``data`` as a chunk of ``category``."""
    target = chunk_class(category)
    try:
        return target.model_validate(data)
    except ValidationError as exc:
        if target is Chunk and category not in _REGISTRY:
            raise ValueError(
                f"category {category!r} is not registered, so its chunks were "
                f"validated as a plain Chunk and rejected. Registered categories: "
                f"{sorted(_REGISTRY)}. If you defined a class for this category, "
                "check it carries the @register_chunk_class decorator; otherwise "
                f"define one pinning category={category!r}.\n{exc}"
            ) from exc
        raise


# --------------------------------------------------------------------------- #
# The category classes -- a CodeChunk *is* a Chunk
# --------------------------------------------------------------------------- #


class LayoutChunk(Chunk):
    """A chunk from a document whose original layout was preserved.

    Which node it was, which page it sat on, and where on that page. **Not a
    category** -- undecorated, so it never registers; it exists to be inherited.

    Named for what it holds rather than for a parser: any layout-aware converter
    produces these. A Markdown-sourced paper inherits it and leaves all three
    empty, using ``start_line``/``end_line`` from the base instead.
    """

    doc_item_refs: list[str] = Field(default_factory=list)  # e.g. "#/texts/104"
    pages: list[int] = Field(default_factory=list)
    provenance: list[PageProvenance] = Field(default_factory=list)


@register_chunk_class
class CodeChunk(Chunk):
    """Source code, or documentation beside it in a repository.

    Inherits :class:`Chunk`, not :class:`LayoutChunk`: a source file has no pages,
    only lines and character offsets.
    """

    category: Literal["code"] = "code"

    symbol: str | None = None  # the function or class this chunk sits in
    is_doc: bool = False  # documentation rather than source

    # Character offsets into the file, where line numbers are not precise enough.
    start_index: int | None = None
    end_index: int | None = None

    @model_validator(mode="after")
    def _index_range_is_coherent(self) -> Self:
        """Validation rule 3, for the character-offset pair."""
        check_range(self.start_index, self.end_index, "start_index", "end_index")
        return self


@register_chunk_class
class PaperChunk(LayoutChunk):
    """A published paper or preprint."""

    category: Literal["paper"] = "paper"

    doi: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    section: str | None = None  # "Methods", "Results"


@register_chunk_class
class ManualChunk(LayoutChunk):
    """An instrument or software manual."""

    category: Literal["manual"] = "manual"

    instrument: str | None = None  # e.g. "MARIA", "KWS-1"
    manual_version: str | None = None


@register_chunk_class
class ReportChunk(LayoutChunk):
    """A technical report, experiment report or internal note."""

    category: Literal["report"] = "report"

    report_number: str | None = None  # how the report is cited
    year: int | None = None


@register_chunk_class
class DatasetChunk(LayoutChunk):
    """A dataset's description or README. The measurement data is not chunked."""

    category: Literal["dataset"] = "dataset"

    doi: str | None = None
    instrument: str | None = None


@register_chunk_class
class PresentationChunk(LayoutChunk):
    """Slides. A slide number is a page number, so it is ``pages`` from
    :class:`LayoutChunk` -- not a second field meaning the same thing.
    """

    category: Literal["presentation"] = "presentation"

    event: str | None = None


@register_chunk_class
class WikiChunk(Chunk):
    """A wiki or documentation-site page.

    Empty on purpose: its address is already ``source_uri`` and its position
    already ``heading_path``. Inherits :class:`Chunk`, not :class:`LayoutChunk` --
    a wiki page is Markdown, with no layout anchors to carry.
    """

    category: Literal["wiki"] = "wiki"


@register_chunk_class
class BookChunk(LayoutChunk):
    """A book, whole or one chapter of it.

    The source's ``title`` is always the book's, however the book was split. The
    chapter's title is the first entry of ``heading_path`` ; ``chapter`` is only
    how it is cited.
    """

    category: Literal["book"] = "book"

    isbn: str | None = None  # ISBN-13, digits only, e.g. "9780198567691"
    chapter: str | None = None  # how the chapter is cited, e.g. "3", "Appendix A"
    authors: list[str] = Field(default_factory=list)  # who writes the book/text
    editors: list[str] = Field(default_factory=list)  # for an edited volume
    year: int | None = None
    publisher: str | None = None
