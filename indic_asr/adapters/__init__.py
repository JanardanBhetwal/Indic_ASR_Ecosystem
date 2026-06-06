"""
indic_asr/adapters/__init__.py
================================
Adapter registry. Maps seed entries to adapter classes automatically.
"""

from __future__ import annotations

import logging
from typing import Optional, Type

from indic_asr.adapters.base import BaseAdapter
from indic_asr.adapters.common_voice import CommonVoiceAdapter
from indic_asr.adapters.fleurs import FLEURSAdapter
from indic_asr.adapters.generic_hf import (
    GenericHFAdapter,
    IndicTTSAdapter,
    ShrutilipiAdapter,
)
from indic_asr.schema import SeedDatasetEntry

logger = logging.getLogger(__name__)

# Ordered list of adapter classes to try (most specific first)
ADAPTER_REGISTRY: list[Type[BaseAdapter]] = [
    CommonVoiceAdapter,
    FLEURSAdapter,
    ShrutilipiAdapter,
    IndicTTSAdapter,
    GenericHFAdapter,  # Always last: generic fallback
]

# Direct lookup by adapter name string (from seed YAML)
ADAPTER_BY_NAME: dict[str, Type[BaseAdapter]] = {
    cls.__name__: cls for cls in ADAPTER_REGISTRY
}


def get_adapter(seed_entry: SeedDatasetEntry) -> Optional[BaseAdapter]:
    """
    Resolve the correct adapter for a seed entry.
    
    Resolution order:
    1. If seed_entry.adapter is a known adapter name, use it directly.
    2. Otherwise, try each adapter in ADAPTER_REGISTRY order via can_handle().
    3. Return None if no adapter matches.
    """
    # Direct name lookup (fastest, most explicit)
    if seed_entry.adapter in ADAPTER_BY_NAME:
        adapter_cls = ADAPTER_BY_NAME[seed_entry.adapter]
        logger.debug(f"Using adapter {adapter_cls.__name__} for {seed_entry.id}")
        return adapter_cls(seed_entry)

    # Auto-detection fallback
    for adapter_cls in ADAPTER_REGISTRY:
        if adapter_cls.can_handle(seed_entry):
            logger.debug(
                f"Auto-selected adapter {adapter_cls.__name__} for {seed_entry.id}"
            )
            return adapter_cls(seed_entry)

    logger.error(f"No adapter found for seed entry {seed_entry.id!r}")
    return None


def list_adapters() -> list[str]:
    """Return names of all registered adapters."""
    return list(ADAPTER_BY_NAME.keys())


__all__ = [
    "BaseAdapter",
    "CommonVoiceAdapter",
    "FLEURSAdapter",
    "GenericHFAdapter",
    "ShrutilipiAdapter",
    "IndicTTSAdapter",
    "get_adapter",
    "list_adapters",
    "ADAPTER_REGISTRY",
]
