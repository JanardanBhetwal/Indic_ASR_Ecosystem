"""
indic_asr — Indic ASR Dataset Ecosystem
"""

__version__ = "0.1.0"

from indic_asr.schema import (
    INDIC_LANGUAGE_CODES,
    RegistryEntry,
    ManifestRow,
    SeedDatasetEntry,
    normalize_language_code,
)

__all__ = [
    "INDIC_LANGUAGE_CODES",
    "RegistryEntry",
    "ManifestRow",
    "SeedDatasetEntry",
    "normalize_language_code",
]
