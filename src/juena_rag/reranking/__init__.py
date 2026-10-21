"""The second stage of a search: a model that re-reads the closest candidates.

:class:`Reranker` is the shape any reranker has, and :class:`RerankFailed` the one
failure a search survives. Both import nothing beyond the standard library; the
real reranker, :mod:`juena_rag.reranking.blablador`, needs the ``[embed]`` extra.
"""

from juena_rag.reranking.base import Reranker, RerankFailed

__all__ = ["RerankFailed", "Reranker"]
