"""Builders shared by the schema tests.

Plain functions rather than fixtures, because they take arguments: a test says
`a_source(visibility="public")` to vary one field and leave the rest alone.
"""

from juena_rag import SourceMetadata


def a_source(**overrides) -> SourceMetadata:
    defaults = {
        "source_id": "kws1-manual",
        "category": "manual",
        "revision": "v3",
        # Required on the model, so the builder has to state it too -- which is
        # the point: no call site gets a permission it did not ask for.
        "visibility": "internal",
    }
    return SourceMetadata(**{**defaults, **overrides})


def a_docling_source(**overrides) -> SourceMetadata:
    return a_source(
        artifact_uri="s3://juena/manuals/kws1-v3.docling.json",
        artifact_format="docling-json",
        **overrides,
    )
