"""The HTTP service: the same `Store`, reachable over the network.

``settings`` needs nothing beyond the standard library, so it imports here.
``app`` is deliberately **not** imported: it needs the ``[serve]`` extra, and this
module has to keep working without it. Reach for it by name::

    from juena_rag.service.app import create_app   # needs [serve]
"""

from juena_rag.service.settings import ServiceSettings

__all__ = ["ServiceSettings"]
