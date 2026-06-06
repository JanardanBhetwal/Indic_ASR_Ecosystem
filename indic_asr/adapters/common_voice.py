"""
indic_asr/adapters/common_voice.py
====================================
Adapter for Mozilla Common Voice datasets (all versions).

Handles:
  - mozilla-foundation/common_voice_6_0 through common_voice_17_0+
  - Config per language (e.g. config="hi" for Hindi)
  - Versioned so schema drift is detected automatically
"""

from __future__ import annotations

import logging
from typing import Iterator, Optional

from indic_asr.adapters.base import BaseAdapter, SchemaIncompatibleError
from indic_asr.schema import (
    AudioFormat,
    AudioStats,
    LicenseType,
    QualityTier,
    RegistryEntry,
    SeedDatasetEntry,
    SplitType,
    SourceType,
    normalize_language_code,
)

logger = logging.getLogger(__name__)


class CommonVoiceAdapter(BaseAdapter):
    """
    Adapter for Mozilla Common Voice (all versions).
    
    Common Voice uses one config per language. The config name
    IS the BCP-47 language code (or close to it).
    
    Canonical splits: train, validation, test
    Audio column: audio
    Transcript column: sentence
    """

    VERSION = "1.2.0"

    # Common Voice license is CC0 (public domain)
    LICENSE = LicenseType.CC0

    # Known CV column schemas (may drift across versions)
    KNOWN_TRANSCRIPT_COLS = ["sentence", "sentence_domain", "raw_filename"]
    KNOWN_AUDIO_COLS = ["audio", "path"]

    # CV-specific Indic language config names (some differ from BCP-47)
    CV_CONFIG_TO_BCP47: dict[str, str] = {
        "hi": "hi",
        "bn": "bn",
        "mr": "mr",
        "ta": "ta",
        "te": "te",
        "kn": "kn",
        "ml": "ml",
        "ur": "ur",
        "gu": "gu",
        "pa-IN": "pa",
        "or": "or",
        "as": "as",
        "ne-NP": "ne",
        "sa": "sa",
    }

    SPLITS_TO_INGEST = [
        (SplitType.TRAIN, "train"),
        (SplitType.VALIDATION, "validation"),
        (SplitType.TEST, "test"),
    ]

    @classmethod
    def can_handle(cls, seed_entry: SeedDatasetEntry) -> bool:
        return (
            seed_entry.adapter == "CommonVoiceAdapter"
            or (
                seed_entry.hf_repo_id is not None
                and "common_voice" in seed_entry.hf_repo_id.lower()
            )
        )

    def _resolve_configs(self, languages: Optional[list[str]]) -> list[tuple[str, str]]:
        """
        Returns list of (cv_config_name, bcp47_code) pairs to process.
        
        If seed_entry.hf_configs is specified, use those.
        Otherwise, use all Indic language configs we know about.
        """
        if self.seed_entry.hf_configs:
            result = []
            for cfg in self.seed_entry.hf_configs:
                bcp47 = self.CV_CONFIG_TO_BCP47.get(cfg)
                if bcp47 is None:
                    # Try direct normalization
                    try:
                        bcp47 = normalize_language_code(cfg)
                    except ValueError:
                        logger.warning(f"Unknown CV config {cfg!r}, skipping")
                        continue
                if languages is None or bcp47 in languages:
                    result.append((cfg, bcp47))
            return result
        else:
            # Use all known Indic configs
            result = []
            for cv_cfg, bcp47 in self.CV_CONFIG_TO_BCP47.items():
                if languages is None or bcp47 in languages:
                    result.append((cv_cfg, bcp47))
            return result

    def iter_registry_entries(
        self,
        languages: Optional[list[str]] = None,
        splits: Optional[list[SplitType]] = None,
        dry_run: bool = False,
    ) -> Iterator[RegistryEntry]:
        
        repo_id = self.seed_entry.hf_repo_id
        if not repo_id:
            raise ValueError(f"CommonVoiceAdapter requires hf_repo_id in seed entry {self.seed_entry.id}")

        # Fetch commit SHA once for the whole repo
        commit_sha = self._get_hf_commit_sha(repo_id) if not dry_run else "dry-run"
        
        configs_to_process = self._resolve_configs(languages)
        
        for cv_config, bcp47 in configs_to_process:
            logger.info(f"Processing {repo_id} / config={cv_config} (lang={bcp47})")
            
            builder = self._safe_load_builder(repo_id, cv_config)
            if builder is None:
                logger.warning(f"Skipping {repo_id}/{cv_config}: builder unavailable")
                continue

            # Detect columns
            try:
                features = dict(builder.info.features) if builder.info.features else {}
                columns = list(features.keys())
                
                audio_col = self._detect_cv_audio_column(columns)
                transcript_col = self._detect_cv_transcript_column(columns)
                schema_hash = self.compute_schema_hash(features)
                
            except SchemaIncompatibleError as e:
                logger.error(f"Schema incompatible for {repo_id}/{cv_config}: {e}")
                continue

            # Process each split
            available_splits = list(builder.info.splits.keys()) if builder.info.splits else []
            
            for canonical_split, source_split_name in self.SPLITS_TO_INGEST:
                if splits and canonical_split not in splits:
                    continue
                if source_split_name not in available_splits:
                    logger.debug(f"Split {source_split_name!r} not in {repo_id}/{cv_config}")
                    continue

                # Get split size
                split_info = builder.info.splits.get(source_split_name)
                num_samples = split_info.num_examples if split_info else 0

                stats = AudioStats(
                    num_samples=num_samples,
                    total_duration_seconds=0.0,  # Expensive; computed by validation pipeline
                    audio_format=AudioFormat.MP3,  # CV uses MP3
                ) if not dry_run else None

                entry_id = self.make_entry_id(bcp47, canonical_split)
                provenance = self.make_provenance(
                    hf_config=cv_config,
                    hf_split=source_split_name,
                    hf_commit_sha=commit_sha,
                    schema_hash=schema_hash,
                )

                yield RegistryEntry(
                    entry_id=entry_id,
                    name=f"Common Voice ({cv_config}) — {canonical_split.value}",
                    description=(
                        f"Mozilla Common Voice crowd-sourced speech dataset "
                        f"for {bcp47}, {canonical_split.value} split."
                    ),
                    language=bcp47,
                    split=canonical_split,
                    domain="read_speech",
                    source_type=SourceType.HUGGINGFACE,
                    hf_repo_id=repo_id,
                    hf_config=cv_config,
                    hf_split_name=source_split_name,
                    hf_audio_column=audio_col,
                    hf_transcript_column=transcript_col,
                    hf_streaming_compatible=True,
                    license=self.LICENSE,
                    quality_tier=QualityTier.SILVER,  # CV is crowd-sourced with validation
                    is_synthetic=False,
                    is_noisy=False,
                    validation_passed=False,  # Will be set by validation pipeline
                    stats=stats,
                    provenance=provenance,
                )

    def _detect_cv_audio_column(self, columns: list[str]) -> str:
        """CV-specific audio column detection."""
        # CV traditionally uses "audio", fallback to "path"
        for col in ["audio", "path"]:
            if col in columns:
                return col
        raise SchemaIncompatibleError(
            f"No CV audio column found in: {columns}"
        )

    def _detect_cv_transcript_column(self, columns: list[str]) -> str:
        """CV-specific transcript column detection."""
        for col in ["sentence", "normalized_text", "raw_text", "text"]:
            if col in columns:
                return col
        raise SchemaIncompatibleError(
            f"No CV transcript column found in: {columns}"
        )
