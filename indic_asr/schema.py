"""
indic_asr/schema.py
====================
Pydantic v2 models for every schema in the ecosystem.

Layer hierarchy:
  SeedDatasetEntry     → Layer 0 (seed_datasets.yaml)
  RegistryEntry        → Layer 2 (registry/entries/*.json)
  ManifestRow          → Layer 3 (manifests/*.parquet)
  ManifestMetadata     → Layer 3 sidecar

All language codes are BCP-47.
All timestamps are ISO-8601 UTC.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class SplitType(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    OTHER = "other"


class AudioFormat(str, Enum):
    MP3 = "mp3"
    WAV = "wav"
    FLAC = "flac"
    OGG = "ogg"
    OPUS = "opus"
    UNKNOWN = "unknown"


class LicenseType(str, Enum):
    CC0 = "cc0-1.0"
    CC_BY = "cc-by-4.0"
    CC_BY_SA = "cc-by-sa-4.0"
    CC_BY_NC = "cc-by-nc-4.0"
    CC_BY_NC_SA = "cc-by-nc-sa-4.0"
    APACHE2 = "apache-2.0"
    MIT = "mit"
    RESEARCH_ONLY = "research-only"
    UNKNOWN = "unknown"
    OTHER = "other"


class SourceType(str, Enum):
    HUGGINGFACE = "huggingface"
    OPENSLR = "openslr"
    CUSTOM_URL = "custom_url"
    LOCAL = "local"


class QualityTier(str, Enum):
    """Tier assigned by validation pipeline."""
    GOLD = "gold"      # Studio-quality, verified transcripts
    SILVER = "silver"  # Good quality, crowd-sourced with validation
    BRONZE = "bronze"  # Acceptable quality, minimal validation
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Canonical language codes (BCP-47 subset for Indic languages)
# ---------------------------------------------------------------------------

INDIC_LANGUAGE_CODES: dict[str, str] = {
    # BCP-47 code → language name
    "hi": "Hindi",
    "bn": "Bengali",
    "as": "Assamese",
    "gu": "Gujarati",
    "mr": "Marathi",
    "pa": "Punjabi",
    "ur": "Urdu",
    "or": "Odia",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
    "ne": "Nepali",
    "sa": "Sanskrit",
    "mai": "Maithili",
    "bho": "Bhojpuri",
    "kok": "Konkani",
    "sd": "Sindhi",
    "si": "Sinhala",
    "dz": "Dzongkha",
}

# Aliases to canonical BCP-47
LANGUAGE_ALIASES: dict[str, str] = {
    "hindi": "hi",
    "hin": "hi",
    "hi-in": "hi",
    "hi_in": "hi",
    "bengali": "bn",
    "ben": "bn",
    "assamese": "as",
    "asm": "as",
    "gujarati": "gu",
    "guj": "gu",
    "marathi": "mr",
    "mar": "mr",
    "punjabi": "pa",
    "pun": "pa",
    "pan": "pa",
    "urdu": "ur",
    "urd": "ur",
    "odia": "or",
    "oriya": "or",
    "ori": "or",
    "tamil": "ta",
    "tam": "ta",
    "telugu": "te",
    "tel": "te",
    "kannada": "kn",
    "kan": "kn",
    "malayalam": "ml",
    "mal": "ml",
    "nepali": "ne",
    "nep": "ne",
    "sanskrit": "sa",
    "san": "sa",
}


def normalize_language_code(raw: str) -> str:
    """Normalize any language code variant to canonical BCP-47."""
    cleaned = raw.strip().lower().replace("-", "_")
    # Try direct lookup first
    if cleaned in INDIC_LANGUAGE_CODES:
        return cleaned
    # Try aliases
    if cleaned in LANGUAGE_ALIASES:
        return LANGUAGE_ALIASES[cleaned]
    # Try stripping region suffix (hi_IN → hi)
    base = cleaned.split("_")[0]
    if base in INDIC_LANGUAGE_CODES:
        return base
    raise ValueError(f"Unknown language code: {raw!r}")


# ---------------------------------------------------------------------------
# Layer 0: Seed catalogue schema
# ---------------------------------------------------------------------------


class SeedDatasetEntry(BaseModel):
    """
    One entry in seed_datasets.yaml.
    Human-maintained. Minimal fields required.
    """

    model_config = {"extra": "forbid"}

    # Unique stable ID for this seed entry
    id: str = Field(
        description="Stable unique identifier, e.g. 'mozilla-foundation/common_voice_17_0'"
    )

    source_type: SourceType = SourceType.HUGGINGFACE

    # HF-specific fields
    hf_repo_id: Optional[str] = Field(
        None,
        description="HuggingFace dataset repo ID, e.g. 'mozilla-foundation/common_voice_17_0'",
    )
    hf_configs: Optional[list[str]] = Field(
        None,
        description="List of HF config names to extract. If None, auto-detect from language.",
    )

    # Language(s) this seed covers
    languages: list[str] = Field(
        description="BCP-47 language codes covered by this seed entry."
    )

    # Adapter to use
    adapter: str = Field(
        description="Adapter class name, e.g. 'CommonVoiceAdapter'"
    )

    # Human-readable metadata
    name: str = Field(description="Human-readable dataset name")
    license: LicenseType = LicenseType.UNKNOWN
    quality_tier: QualityTier = QualityTier.UNKNOWN
    notes: Optional[str] = None
    enabled: bool = True

    @field_validator("languages", mode="before")
    @classmethod
    def normalize_languages(cls, v: list[str]) -> list[str]:
        return [normalize_language_code(lang) for lang in v]


# ---------------------------------------------------------------------------
# Layer 2: Registry entry schema
# ---------------------------------------------------------------------------


class AudioStats(BaseModel):
    """Aggregate audio statistics for a dataset split."""

    model_config = {"extra": "allow"}

    num_samples: int
    total_duration_seconds: float
    mean_duration_seconds: Optional[float] = None
    std_duration_seconds: Optional[float] = None
    min_duration_seconds: Optional[float] = None
    max_duration_seconds: Optional[float] = None
    sample_rate_hz: Optional[int] = None
    audio_format: AudioFormat = AudioFormat.UNKNOWN

    # Transcript stats
    mean_transcript_length_chars: Optional[float] = None
    vocabulary_size: Optional[int] = None

    # Quality signals
    snr_mean_db: Optional[float] = None
    snr_std_db: Optional[float] = None


class ProvenanceRecord(BaseModel):
    """
    Full provenance chain for a registry entry.
    Immutable once written.
    """

    model_config = {"extra": "forbid"}

    # Source identification
    source_type: SourceType
    source_id: str = Field(description="Canonical source ID from seed catalogue")
    source_url: Optional[str] = None

    # HF-specific provenance
    hf_repo_id: Optional[str] = None
    hf_config: Optional[str] = None
    hf_split: Optional[str] = None
    hf_commit_sha: Optional[str] = Field(
        None,
        description="Git commit SHA of the HF dataset repo at ingest time"
    )
    hf_dataset_version: Optional[str] = None

    # Adapter used
    adapter_class: str
    adapter_version: str

    # Ingest metadata
    ingested_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    ingested_by: str = Field(description="Tool or person that ran ingestion")

    # Content fingerprint
    schema_hash: Optional[str] = Field(
        None,
        description="SHA256 of the source dataset schema at ingest time"
    )


class RegistryEntry(BaseModel):
    """
    One registry entry = one (dataset, language, split) triple.
    Written by the ingest pipeline, validated before commit.
    """

    model_config = {"extra": "forbid"}

    # Primary key
    entry_id: str = Field(
        description="Stable unique ID: {source_id}__{lang}__{split}, "
                    "e.g. 'common_voice_17__hi__train'"
    )

    # Human-readable
    name: str
    description: Optional[str] = None

    # Classification
    language: str = Field(description="BCP-47 language code")
    split: SplitType
    domain: Optional[str] = Field(
        None,
        description="e.g. 'read_speech', 'spontaneous', 'telephone', 'broadcast'"
    )

    # Source and access
    source_type: SourceType
    hf_repo_id: Optional[str] = None
    hf_config: Optional[str] = None
    hf_split_name: Optional[str] = Field(
        None,
        description="Actual split name in the source (may differ from canonical split)"
    )
    hf_audio_column: str = "audio"
    hf_transcript_column: str = "sentence"
    hf_streaming_compatible: bool = True

    # Quality and licensing
    license: LicenseType
    quality_tier: QualityTier
    is_synthetic: bool = False
    is_noisy: bool = False
    validation_passed: bool = False

    # Statistics (populated by validation pipeline)
    stats: Optional[AudioStats] = None

    # Deduplication fingerprints
    content_fingerprints: Optional[list[str]] = Field(
        None,
        description="Sample of audio content hashes for cross-dataset dedup"
    )

    # Provenance (immutable)
    provenance: ProvenanceRecord

    # Schema version for forward compatibility
    schema_version: str = "1.0"

    @field_validator("language", mode="before")
    @classmethod
    def normalize_lang(cls, v: str) -> str:
        return normalize_language_code(v)

    @field_validator("entry_id")
    @classmethod
    def validate_entry_id(cls, v: str) -> str:
        if not all(c.isalnum() or c in "_-." for c in v):
            raise ValueError(f"entry_id must be alphanumeric with _ - . only: {v!r}")
        return v

    def to_registry_path(self) -> str:
        """Returns relative path for this entry in the registry directory."""
        return f"registry/entries/{self.entry_id}.json"


# ---------------------------------------------------------------------------
# Layer 3: Manifest row schema
# ---------------------------------------------------------------------------


class ManifestRow(BaseModel):
    """
    One row in a manifest Parquet file.
    This is what model training pipelines consume directly.
    
    Design principle: every field a modern ASR trainer needs,
    nothing more. Audio is always a pointer, never inline bytes.
    """

    model_config = {"extra": "forbid"}

    # Unique identifier for this utterance
    utterance_id: str = Field(
        description="Globally unique utterance ID: {entry_id}__{row_index}"
    )

    # Audio pointer (streaming-compatible)
    audio_hf_repo: str = Field(description="HF repo ID where audio lives")
    audio_hf_config: Optional[str] = None
    audio_hf_split: str
    audio_hf_row_index: int = Field(
        description="Row index in the source HF dataset for exact reproducibility"
    )

    # Transcript
    transcript: str
    transcript_normalized: Optional[str] = Field(
        None,
        description="Normalized transcript (lowercased, punctuation-stripped) "
                    "for acoustic model training"
    )

    # Language
    language: str  # BCP-47

    # Audio properties (populated during validation)
    duration_seconds: Optional[float] = None
    sample_rate_hz: Optional[int] = None
    audio_format: Optional[str] = None

    # Provenance back-pointer
    source_entry_id: str = Field(
        description="Registry entry_id this row came from"
    )
    source_split: SplitType

    # Quality signals
    quality_tier: QualityTier
    snr_db: Optional[float] = None
    is_deduplicated: bool = True

    # Content fingerprint for deduplication
    audio_fingerprint: Optional[str] = Field(
        None,
        description="SHA256 of first 2 seconds of audio waveform bytes (lightweight)"
    )

    # License propagation
    license: LicenseType

    @field_validator("language", mode="before")
    @classmethod
    def normalize_lang(cls, v: str) -> str:
        return normalize_language_code(v)


class ManifestMetadata(BaseModel):
    """
    Sidecar metadata for a manifest file.
    Stored alongside the Parquet as _metadata.json.
    """

    manifest_id: str
    language: str
    split: SplitType
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    generator_version: str
    num_rows: int
    total_duration_seconds: float
    source_entry_ids: list[str]
    filters_applied: dict[str, Any] = Field(default_factory=dict)
    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
