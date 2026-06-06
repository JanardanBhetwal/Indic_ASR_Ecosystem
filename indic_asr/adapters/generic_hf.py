"""
indic_asr/adapters/generic_hf.py
==================================
Generic adapter for well-structured single-config HF datasets.
Also includes the ShrutilipiAdapter for AI4Bharat's Shrutilipi corpus.

Use GenericHFAdapter when:
  - Dataset has one config per language (or one config for one language)
  - Standard audio/transcript column names
  - Standard splits (train/validation/test)
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


class GenericHFAdapter(BaseAdapter):
    """
    Fallback adapter for HuggingFace datasets with standard structure.
    
    Suitable for datasets where:
    - One HF config covers one language
    - Audio and transcript columns have recognizable names
    - No major schema quirks
    
    The seed entry MUST specify hf_configs if the dataset has configs,
    and MUST list the exact languages covered.
    """

    VERSION = "1.0.0"

    SPLITS_TO_INGEST = [
        (SplitType.TRAIN, "train"),
        (SplitType.VALIDATION, "validation"),
        (SplitType.TEST, "test"),
    ]
    SPLIT_FALLBACKS = {
        "validation": ["dev", "valid", "val"],
        "test": ["eval", "test_clean"],
        "train": ["training", "train_clean"],
    }

    @classmethod
    def can_handle(cls, seed_entry: SeedDatasetEntry) -> bool:
        return seed_entry.adapter == "GenericHFAdapter"

    def iter_registry_entries(
        self,
        languages: Optional[list[str]] = None,
        splits: Optional[list[SplitType]] = None,
        dry_run: bool = False,
    ) -> Iterator[RegistryEntry]:

        repo_id = self.seed_entry.hf_repo_id
        if not repo_id:
            raise ValueError("GenericHFAdapter requires hf_repo_id")

        commit_sha = self._get_hf_commit_sha(repo_id) if not dry_run else "dry-run"

        # Determine configs to process
        configs_to_process: list[tuple[Optional[str], str]]
        
        if self.seed_entry.hf_configs:
            # One config per language assumed
            if len(self.seed_entry.hf_configs) != len(self.seed_entry.languages):
                # Multi-config, multi-language: zip them
                # Or configs cover multiple languages (less common)
                logger.warning(
                    f"{repo_id}: hf_configs and languages count mismatch. "
                    "Processing each config for all languages."
                )
                configs_to_process = [
                    (cfg, lang)
                    for cfg in self.seed_entry.hf_configs
                    for lang in self.seed_entry.languages
                    if languages is None or lang in languages
                ]
            else:
                configs_to_process = [
                    (cfg, lang)
                    for cfg, lang in zip(self.seed_entry.hf_configs, self.seed_entry.languages)
                    if languages is None or lang in languages
                ]
        else:
            # No config: dataset is single-config
            configs_to_process = [
                (None, lang)
                for lang in self.seed_entry.languages
                if languages is None or lang in languages
            ]

        for hf_config, bcp47 in configs_to_process:
            logger.info(f"Processing {repo_id} config={hf_config} lang={bcp47}")

            builder = self._safe_load_builder(repo_id, hf_config)
            if builder is None:
                continue

            try:
                features = dict(builder.info.features) if builder.info.features else {}
                columns = list(features.keys())
                audio_col = self.detect_audio_column(columns)
                transcript_col = self.detect_transcript_column(columns)
                schema_hash = self.compute_schema_hash(features)
            except SchemaIncompatibleError as e:
                logger.error(f"Schema error for {repo_id}/{hf_config}: {e}")
                continue

            available_splits = (
                list(builder.info.splits.keys()) if builder.info.splits else []
            )

            for canonical_split, source_split_name in self.SPLITS_TO_INGEST:
                if splits and canonical_split not in splits:
                    continue

                # Try primary split name, then fallbacks
                resolved_split = self._resolve_split(
                    source_split_name, canonical_split, available_splits
                )
                if resolved_split is None:
                    logger.debug(
                        f"No split found for {canonical_split.value} in {repo_id}"
                    )
                    continue

                split_info = builder.info.splits.get(resolved_split)
                num_samples = split_info.num_examples if split_info else 0

                stats = AudioStats(
                    num_samples=num_samples,
                    total_duration_seconds=0.0,
                    audio_format=AudioFormat.UNKNOWN,
                ) if not dry_run else None

                entry_id = self.make_entry_id(bcp47, canonical_split)
                provenance = self.make_provenance(
                    hf_config=hf_config,
                    hf_split=resolved_split,
                    hf_commit_sha=commit_sha,
                    schema_hash=schema_hash,
                )

                yield RegistryEntry(
                    entry_id=entry_id,
                    name=f"{self.seed_entry.name} ({bcp47}) — {canonical_split.value}",
                    language=bcp47,
                    split=canonical_split,
                    source_type=SourceType.HUGGINGFACE,
                    hf_repo_id=repo_id,
                    hf_config=hf_config,
                    hf_split_name=resolved_split,
                    hf_audio_column=audio_col,
                    hf_transcript_column=transcript_col,
                    hf_streaming_compatible=True,
                    license=self.seed_entry.license,
                    quality_tier=self.seed_entry.quality_tier,
                    is_synthetic=False,
                    is_noisy=False,
                    validation_passed=False,
                    stats=stats,
                    provenance=provenance,
                )

    def _resolve_split(
        self,
        primary: str,
        canonical: SplitType,
        available: list[str],
    ) -> Optional[str]:
        if primary in available:
            return primary
        fallbacks = self.SPLIT_FALLBACKS.get(primary, [])
        for fallback in fallbacks:
            if fallback in available:
                logger.debug(f"Using fallback split {fallback!r} for {canonical.value}")
                return fallback
        return None


class ShrutilipiAdapter(GenericHFAdapter):
    """
    Adapter for AI4Bharat Shrutilipi corpus.
    
    Shrutilipi is a large-scale, broadcast news-domain ASR dataset
    covering 12 Indian languages.
    
    HF repo: ai4bharat/shrutilipi
    """

    VERSION = "1.0.0"

    @classmethod
    def can_handle(cls, seed_entry: SeedDatasetEntry) -> bool:
        return (
            seed_entry.adapter == "ShrutilipiAdapter"
            or (
                seed_entry.hf_repo_id is not None
                and "shrutilipi" in seed_entry.hf_repo_id.lower()
            )
        )

    def iter_registry_entries(
        self,
        languages: Optional[list[str]] = None,
        splits: Optional[list[SplitType]] = None,
        dry_run: bool = False,
    ) -> Iterator[RegistryEntry]:
        # Shrutilipi has broadcast domain speech
        for entry in super().iter_registry_entries(languages, splits, dry_run):
            # Override domain and quality
            yield RegistryEntry(
                **{
                    **entry.model_dump(),
                    "domain": "broadcast_news",
                    "quality_tier": QualityTier.SILVER,
                    "description": (
                        f"AI4Bharat Shrutilipi broadcast news ASR corpus "
                        f"for {entry.language}, {entry.split.value} split."
                    ),
                }
            )


class IndicTTSAdapter(GenericHFAdapter):
    """
    Adapter for AI4Bharat IndicTTS studio-recorded corpus.
    Single speaker, studio quality.
    """

    VERSION = "1.0.0"

    @classmethod
    def can_handle(cls, seed_entry: SeedDatasetEntry) -> bool:
        return seed_entry.adapter == "IndicTTSAdapter"

    def iter_registry_entries(
        self,
        languages: Optional[list[str]] = None,
        splits: Optional[list[SplitType]] = None,
        dry_run: bool = False,
    ) -> Iterator[RegistryEntry]:
        for entry in super().iter_registry_entries(languages, splits, dry_run):
            yield RegistryEntry(
                **{
                    **entry.model_dump(),
                    "domain": "studio_read_speech",
                    "quality_tier": QualityTier.GOLD,
                    "description": (
                        f"AI4Bharat IndicTTS studio-recorded speech for {entry.language}."
                    ),
                }
            )
