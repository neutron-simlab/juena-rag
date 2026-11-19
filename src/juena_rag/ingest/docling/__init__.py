"""Docling: a HybridChunker export in, :class:`BookChunk` rows out, one export per source.

- :mod:`~juena_rag.ingest.docling.chunking`: one chunker record into a book chunk
- :mod:`~juena_rag.ingest.docling.manifest`: the CSV listing exports to ingest
- :mod:`~juena_rag.ingest.docling.pipeline`: wires them to the store;
  ``juena-rag ingest-docling`` and ``ingest-docling-csv`` call it

Nothing is imported here, for the same reason as in :mod:`juena_rag.ingest`.
"""
