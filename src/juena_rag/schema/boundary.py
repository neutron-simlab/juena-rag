"""What crosses into the store from outside: a batch going in, a query coming to it.

:class:`ChunkBatch` is the only thing an ingestion pipeline has to build, and it
is where the four rules needing to see *both* are checked -- a chunk cannot tell
on its own whether its source names an artifact. :class:`SearchRequest` is the
other direction and much smaller: the arguments of ``Store.search``, as an object,
because over HTTP they arrive as somebody else's JSON.
"""

from typing import Any, Literal, Self

from pydantic import BaseModel, Field, SerializeAsAny, model_validator

from juena_rag.schema.models import Chunk, SourceMetadata
from juena_rag.schema.registry import chunk_class, parse_chunk


def _coerce_chunk(value: Any, target: type[Chunk], category: str, index: int) -> Any:
    """Resolve one entry of ``ChunkBatch.chunks`` against the batch's category."""
    if isinstance(value, Chunk):
        if isinstance(value, target):
            return value
        if type(value) is Chunk:
            # Rule 4: a bare Chunk is upgraded, not rejected. It holds only base
            # fields, which the category class also has, so nothing is lost.
            return target.model_validate(value.model_dump())
        raise ValueError(
            f"chunk {index} is a {type(value).__name__}, but the batch declares "
            f"category={category!r}, whose class is {target.__name__}"
        )
    if isinstance(value, dict):
        return parse_chunk(category, value)
    return value


class ChunkBatch(BaseModel):
    """A complete statement about one source: its metadata and its chunks."""

    model_config = {"extra": "forbid"}

    source: SourceMetadata

    #: ``SerializeAsAny`` is load-bearing. Pydantic serialises through the
    #: *declared* type, so a plain ``list[Chunk]`` would drop every
    #: category-specific field on the way to the database, silently.
    chunks: list[SerializeAsAny[Chunk]]

    mode: Literal["replace", "upsert"] = "replace"

    @model_validator(mode="before")
    @classmethod
    def _resolve_chunks_against_category(cls, data: Any) -> Any:
        """Build every chunk as the class the source's category names.

        One class for the whole batch, since a batch has one source and therefore
        one category. Without this, a batch parsed from JSON would produce bare
        ``Chunk`` objects and ``extra="forbid"`` would reject every extra key.
        """
        if not isinstance(data, dict):
            return data

        source = data.get("source")
        if isinstance(source, SourceMetadata):
            category = source.category
        elif isinstance(source, dict):
            category = source.get("category")
        else:
            return data
        if not isinstance(category, str):
            return data

        chunks = data.get("chunks")
        if not isinstance(chunks, (list, tuple)):
            return data

        target = chunk_class(category)
        return {
            **data,
            "chunks": [
                _coerce_chunk(chunk, target, category, index)
                for index, chunk in enumerate(chunks)
            ],
        }

    @model_validator(mode="after")
    def _batch_agrees_with_source(self) -> Self:
        """Validation rules 5, 6 and 7."""
        # Rule 7. The atomic replace inserts, moves active_revision, then deletes
        # every other revision -- so an empty batch deletes the source's content
        # and reports success. delete_source() exists for the deliberate case.
        if self.mode == "replace" and not self.chunks:
            raise ValueError(
                f"mode='replace' with no chunks would delete every chunk of source "
                f"{self.source.source_id!r} and report success; use delete_source() "
                "to remove a source deliberately"
            )

        for index, chunk in enumerate(self.chunks):
            refs: list[str] = getattr(chunk, "doc_item_refs", None) or []
            pages: list[int] = getattr(chunk, "pages", None) or []

            # Rule 5. Coordinates into a file require the file to be named.
            if (refs or pages) and not self.source.artifact_uri:
                carried = " and ".join(
                    name
                    for name, is_set in (("doc_item_refs", refs), ("pages", pages))
                    if is_set
                )
                raise ValueError(
                    f"chunk {index} carries {carried} but the source names no "
                    "artifact_uri, so the coordinates point at nothing"
                )

            # Rule 6. Node addresses exist only in a parsed document tree.
            if refs and self.source.artifact_format != "docling-json":
                raise ValueError(
                    f"chunk {index} sets doc_item_refs but artifact_format is "
                    f"{self.source.artifact_format!r}. An address like {refs[0]!r} "
                    "names one entry of a docling-json tree; a markdown artifact is "
                    "a flat file with no addressable entries, so record "
                    "start_line/end_line instead"
                )

        return self


class SearchRequest(BaseModel):
    """One question and the filters to ask it under -- the body of ``POST /v1/search``.

    Every field is named for the ``Store.search`` argument it becomes, so the JSON
    and the Python call read the same. Defined here rather than in the service so
    that ``HttpStore`` can build one without installing a web framework.
    """

    # A misspelled filter is refused rather than dropped. Without this, `{"source":
    # ["vitess"]}` -- singular, which is the natural typo -- would search the whole
    # corpus and return a plausible answer with no filter applied and no error.
    model_config = {"extra": "forbid"}

    query: str = Field(min_length=1)

    sources: list[str] | None = None
    categories: list[str] | None = None
    chunk_kinds: list[str] | None = None

    #: How many candidates the index retrieves, and how many survive to the caller.
    #: Left unset, the service's own defaults apply -- a caller does not have to
    #: know them, and cannot be given a stale copy of them.
    k: int | None = Field(default=None, gt=0)
    n: int | None = Field(default=None, gt=0)

    #: Re-order the closest candidates with the service's reranking model. Off
    #: unless the caller asks: it costs extra model calls on every search.
    rerank: bool = False
