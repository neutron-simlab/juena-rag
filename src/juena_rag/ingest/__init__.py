"""Ingestion pipelines that ship with the package. Needs the ``[ingest]`` extra.

One subpackage per pipeline, named for the kind of artifact it reads:

- :mod:`juena_rag.ingest.code`, run as ``juena-rag ingest-repo``. Clones a git
  repository, splits its files with the sizes the caller's ``repositories.yaml``
  sets, embeds the chunks and stores them as one source.
- :mod:`juena_rag.ingest.docling`, run as ``juena-rag ingest-docling``. Adapts
  one rag-lab HybridChunker export (``*_raw_chunks.json``) into book chunks,
  embeds them and stores them as one source; ``juena-rag ingest-docling-csv``
  does the same for every export a CSV lists.

Nothing is imported here, so ``import juena_rag.ingest`` works on the base
install and only the modules below need the extra.
"""
