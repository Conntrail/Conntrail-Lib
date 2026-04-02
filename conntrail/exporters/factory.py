"""
Exporter factory with module-level caching.

Provides a centralized way to create and cache exporters based on configuration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conntrail.config import ConntrailConfig
    from conntrail.exporters.base import BaseExporter

# Module-level cache: config signature -> exporter instance
_exporter_cache: dict[str, "BaseExporter"] = {}


def get_exporter(config: "ConntrailConfig") -> "BaseExporter":
    """Get or create an exporter based on config.

    Uses a module-level cache to reuse exporters across interceptors
    with the same configuration.

    Args:
        config: ConntrailConfig containing export_format and export_path.

    Returns:
        A cached or newly created BaseExporter instance.
    """
    # Create a cache key from the config attributes that affect exporter creation
    cache_key = f"{config.export_format}:{config.export_path}"

    if cache_key not in _exporter_cache:
        _exporter_cache[cache_key] = _create_exporter(config)

    return _exporter_cache[cache_key]


def _create_exporter(config: "ConntrailConfig") -> "BaseExporter":
    """Create an appropriate exporter based on config.export_format.

    Args:
        config: ConntrailConfig containing export_format and export_path.

    Returns:
        A new BaseExporter instance.
    """
    fmt = config.export_format

    if fmt == "stdout":
        from conntrail.exporters.stdout import StdoutExporter

        return StdoutExporter()
    elif fmt == "jsonl":
        from conntrail.exporters.jsonl import JsonlExporter

        return JsonlExporter(config.export_path)
    else:
        # langsmith — Phase 7, fall back to stdout in the meantime
        from conntrail.exporters.stdout import StdoutExporter

        return StdoutExporter()
