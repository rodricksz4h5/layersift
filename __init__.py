"""layersift plugin — exact, local reading of saved container images.

Registers three read-only tools (``image_summary``, ``image_layer``,
``image_compare``). Pure standard library: no network, no Docker daemon, no
registry, no credentials, no subprocess, no background work, no self-update.
"""

from __future__ import annotations

from . import layersift as _ls


def register(ctx) -> None:
    """Called once by the plugin loader; register the three read-only image tools."""
    ctx.register_tool(name="image_summary", toolset="layersift", schema=_ls.IMAGE_SUMMARY, handler=_ls.image_summary)
    ctx.register_tool(name="image_layer", toolset="layersift", schema=_ls.IMAGE_LAYER, handler=_ls.image_layer)
    ctx.register_tool(name="image_compare", toolset="layersift", schema=_ls.IMAGE_COMPARE, handler=_ls.image_compare)
