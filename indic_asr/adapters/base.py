"""
indic_asr/adapters/base.py
===========================
Abstract base class for all dataset adapters.

Each adapter is responsible for exactly one thing:
translating a source dataset's idiosyncratic schema into
a normalized RegistryEntry (+ optional ManifestRows).

Adapters should:
  - Be stateless
  - Be versioned (bumping version invalidates cached registry entries)
  - Never download audio (only inspect schema/metadata)
  - Be unit-testable without network access via mock fixtures
"""

from __future__ import annotations

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Iterator, Optional

from indic_asr.schema import (
    AudioFormat,
    AudioStats,
    LicenseType,
    ProvenanceRecord,
    QualityTier,
    RegistryEntry,
    SeedDatasetEntry,
    SplitType,
    SourceType,
    normalize_language_code,
)

logger = logging.getLogger(__name__)


class AdapterError(Exception):
    """Raised when an adapter cannot process a dataset."""
    pass


class SchemaIncompatibleError(AdapterError):
    """Raised when the source dataset schema is incompatible with this adapter."""
    pass


class BaseAdapter(ABC):
    """
    Abstract base for all dataset adapters.
    
    Subclasses must implement:
      - VERSION: str class attribute
      - can_handle(): class method
      - iter_registry_entries(): main entry point
    
    Subclasses may optionally override:
      - detect_audio_column()
      - detect_transcript_column()
      - detect_split_mapping()
      - compute_stats()
    """

    VERSION: str  # Must be overridden by each adapter

    # Known column name variants for audio and transcripts
    AUDIO_COLUMN_VARIANTS: list[str] = [
        "audio", "speech", "audio_path", "wav", "sound"
    ]
    TRANSCRIPT_COLUMN_VARIANTS: list[str] = [
        "sentence", "transcription", "text", "transcript",
        "normalized_text", "raw_text", "utterance", "label"
    ]

    def __init__(self, seed_entry: SeedDatasetEntry):
        self.seed_entry = seed_entry
        self._validate_adapter_version()

    def _validate_adapter_version(self) -> None:
        if not hasattr(self, "VERSION"):
            raise NotImplementedError(
                f"Adapter {self.__class__.__name__} must define a VERSION attribute"
            )

    @classmethod
    @abstractmethod
    def can_handle(cls, seed_entry: SeedDatasetEntry) -> bool:
        """
        Return True if this adapter can handle the given seed entry.
        Used for automatic adapter selection.
        """
        raise NotImplementedError

    @abstractmethod
    def iter_registry_entries(
        self,
        languages: Optional[list[str]] = None,
        splits: Optional[list[SplitType]] = None,
        dry_run: bool = False,
    ) -> Iterator[RegistryEntry]:
        """
        Yield one RegistryEntry per (language, split) combination.
        
        Args:
            languages: Filter to specific languages. None = all.
            splits: Filter to specific splits. None = all.
            dry_run: If True, skip expensive stat computation.
        """
        raise NotImplementedError

    def detect_audio_column(self, columns: list[str]) -> str:
        """Find the audio column in a dataset's features."""
        for variant in self.AUDIO_COLUMN_VARIANTS:
            if variant in columns:
                return variant
        raise SchemaIncompatibleError(
            f"Cannot find audio column in {columns}. "
            f"Known variants: {self.AUDIO_COLUMN_VARIANTS}"
        )

    def detect_transcript_column(self, columns: list[str]) -> str:
        """Find the transcript column in a dataset's features."""
        for variant in self.TRANSCRIPT_COLUMN_VARIANTS:
            if variant in columns:
                return variant
        raise SchemaIncompatibleError(
            f"Cannot find transcript column in {columns}. "
            f"Known variants: {self.TRANSCRIPT_COLUMN_VARIANTS}"
        )

    def normalize_split(self, raw_split: str) -> SplitType:
        """Normalize source split names to canonical SplitType."""
        mapping = {
            "train": SplitType.TRAIN,
            "training": SplitType.TRAIN,
            "train+validation": SplitType.TRAIN,
            "dev": SplitType.VALIDATION,
            "valid": SplitType.VALIDATION,
            "validation": SplitType.VALIDATION,
            "val": SplitType.VALIDATION,
            "dev_clean": SplitType.VALIDATION,
            "test": SplitType.TEST,
            "eval": SplitType.TEST,
            "test_clean": SplitType.TEST,
        }
        return mapping.get(raw_split.lower(), SplitType.OTHER)

    def make_entry_id(self, language: str, split: SplitType) -> str:
        """
        Generate a stable entry_id.
        Uses source_id to ensure uniqueness across the registry.
        """
        # Replace slashes and special chars in source_id for filesystem safety
        safe_id = self.seed_entry.id.replace("/", "__").replace(".", "_")
        return f"{safe_id}__{language}__{split.value}"

    def compute_schema_hash(self, features: dict[str, Any]) -> str:
        """Hash the dataset schema for drift detection."""
        canonical = json.dumps(
            {k: str(v) for k, v in sorted(features.items())},
            sort_keys=True
        )
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def make_provenance(
        self,
        hf_config: Optional[str] = None,
        hf_split: Optional[str] = None,
        hf_commit_sha: Optional[str] = None,
        schema_hash: Optional[str] = None,
        source_url: Optional[str] = None,
    ) -> ProvenanceRecord:
        """Build a ProvenanceRecord from common fields."""
        return ProvenanceRecord(
            source_type=self.seed_entry.source_type,
            source_id=self.seed_entry.id,
            source_url=source_url or (
                f"https://huggingface.co/datasets/{self.seed_entry.hf_repo_id}"
                if self.seed_entry.hf_repo_id else None
            ),
            hf_repo_id=self.seed_entry.hf_repo_id,
            hf_config=hf_config,
            hf_split=hf_split,
            hf_commit_sha=hf_commit_sha,
            adapter_class=self.__class__.__name__,
            adapter_version=self.VERSION,
            ingested_at=datetime.now(timezone.utc),
            ingested_by="indic-asr-ingest-pipeline",
            schema_hash=schema_hash,
        )

    def _safe_load_builder(self, repo_id: str, config: Optional[str] = None):
        """
        Safely load a HF dataset builder without downloading data.
        Returns None if the dataset is inaccessible.
        """
        try:
            from datasets import load_dataset_builder
            try:
                builder = load_dataset_builder(repo_id, config, trust_remote_code=False)
            except TypeError:
                # Parquet-native datasets (no custom loading script) use ParquetConfig
                # internally, which does not accept trust_remote_code. Fall back without it.
                builder = load_dataset_builder(repo_id, config)
            return builder
        except Exception as e:
            logger.warning(f"Could not load builder for {repo_id}/{config}: {e}")
            return None

    def _get_hf_commit_sha(self, repo_id: str) -> Optional[str]:
        """Fetch the latest commit SHA for a HF dataset repo."""
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            commits = api.list_repo_commits(repo_id, repo_type="dataset")
            if commits:
                return commits[0].commit_id
        except Exception as e:
            logger.debug(f"Could not fetch commit SHA for {repo_id}: {e}")
        return None
