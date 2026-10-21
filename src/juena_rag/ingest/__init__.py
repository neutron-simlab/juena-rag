"""Ingestion pipelines that ship with the package. Needs the ``[ingest]`` extra.

Today there is one: :mod:`juena_rag.ingest.code_repository`, run as
``juena-rag ingest-repo``. It clones a git repository, splits its files with the
sizes the caller's ``repositories.yaml`` sets, embeds the chunks and stores them
as one source.

Nothing is imported here, so ``import juena_rag.ingest`` works on the base
install and only the modules below need the extra.
"""
