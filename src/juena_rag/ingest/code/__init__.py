"""Code: a git repository in, :class:`CodeChunk` rows out, one source per repository.

- :mod:`~juena_rag.ingest.code.repository`: clone or refresh, pick the files
- :mod:`~juena_rag.ingest.code.chunking`: one file into chunks
- :mod:`~juena_rag.ingest.code.pipeline`: wires them to the store;
  ``juena-rag ingest-repo`` calls it

Nothing is imported here, for the same reason as in :mod:`juena_rag.ingest`.
"""
