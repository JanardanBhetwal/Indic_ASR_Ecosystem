"""
indic_asr/adapters/fleurs.py
==============================
Adapter for Google FLEURS (Few-shot Learning Evaluation of Universal Representations
of Speech) dataset.

FLEURS uses a different config naming convention: "{lang_code}_*"
e.g. "hi_in" for Hindi, "ta_in" for Tamil, etc.

HF repo: google/fleurs
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


class FLEURSAdapter(BaseAdapter):
    """
    Adapter for Google FLEURS multilingual speech dataset.
    
    FLEURS config names use underscore format: hi_in, ta_in, etc.
    Splits: train, validation, test
    Audio column: audio
    Transcript column: transcription (raw) or raw_transcription
    """

    VERSION = "1.1.0"
    LICENSE = LicenseType.CC_BY  # CC-BY 4.0

    # FLEURS config name → BCP-47 mapping for Indic languages
    FLEURS_CONFIG_TO_BCP47: dict[str, str] = {
        "hi_in": "hi",
        "bn_in": "bn",
        "as_in": "as",
        "gu_in": "gu",
        "mr_in": "mr",
        "pa_in": "pa",
        "ur_pk": "ur",
        "or_in": "or",
        "ta_in": "ta",
        "te_in": "te",
        "kn_in": "kn",
        "ml_in": "ml",
        "ne_np": "ne",
        "sd_in": "sd",
    }

    SPLITS_TO_INGEST = [
        (SplitType.TRAIN, "train"),
        (SplitType.VALIDATION, "validation"),
        (SplitType.TEST, "test"),
    ]

    @classmethod
    def can_handle(cls, seed_entry: SeedDatasetEntry) -> bool:
        return (
            seed_entry.adapter == "FLEURSAdapter"
            or (
                seed_entry.hf_repo_id is not None
                and "fleurs" in seed_entry.hf_repo_id.lower()
            )
        )

    def _resolve_configs(self, languages: Optional[list[str]]) -> list[tuple[str, str]]:
        if self.seed_entry.hf_configs:
            result = []
            for cfg in self.seed_entry.hf_configs:
                bcp47 = self.FLEURS_CONFIG_TO_BCP47.get(cfg)
                if bcp47 is None:
                    logger.warning(f"Unknown FLEURS config {cfg!r}, skipping")
                    continue
                if languages is None or bcp47 in languages:
                    result.append((cfg, bcp47))
            return result
        else:
            return [
                (cfg, bcp47)
                for cfg, bcp47 in self.FLEURS_CONFIG_TO_BCP47.items()
                if languages is None or bcp47 in languages
            ]

    def iter_registry_entries(
        self,
        languages: Optional[list[str]] = None,
        splits: Optional[list[SplitType]] = None,
        dry_run: bool = False,
    ) -> Iterator[RegistryEntry]:

        repo_id = self.seed_entry.hf_repo_id
        if not repo_id:
            raise ValueError("FLEURSAdapter requires hf_repo_id")

        commit_sha = self._get_hf_commit_sha(repo_id) if not dry_run else "dry-run"
        configs_to_process = self._resolve_configs(languages)

        for fleurs_config, bcp47 in configs_to_process:
            logger.info(f"Processing FLEURS config={fleurs_config} (lang={bcp47})")

            builder = self._safe_load_builder(repo_id, fleurs_config)
            if builder is None:
                logger.warning(f"Skipping FLEURS/{fleurs_config}: builder unavailable")
                continue

            try:
                features = dict(builder.info.features) if builder.info.features else {}
                columns = list(features.keys())
                audio_col = self.detect_audio_column(columns)
                transcript_col = self._detect_fleurs_transcript_col(columns)
                schema_hash = self.compute_schema_hash(features)
            except SchemaIncompatibleError as e:
                logger.error(f"Schema error for FLEURS/{fleurs_config}: {e}")
                continue

            available_splits = list(builder.info.splits.keys()) if builder.info.splits else []

            for canonical_split, source_split_name in self.SPLITS_TO_INGEST:
                if splits and canonical_split not in splits:
                    continue
                if source_split_name not in available_splits:
                    continue

                split_info = builder.info.splits.get(source_split_name)
                num_samples = split_info.num_examples if split_info else 0

                stats = AudioStats(
                    num_samples=num_samples,
                    total_duration_seconds=0.0,
                    # FLEURS provides 16kHz audio in various formats
                    sample_rate_hz=16000,
                    audio_format=AudioFormat.WAV,
                ) if not dry_run else None

                entry_id = self.make_entry_id(bcp47, canonical_split)
                provenance = self.make_provenance(
                    hf_config=fleurs_config,
                    hf_split=source_split_name,
                    hf_commit_sha=commit_sha,
                    schema_hash=schema_hash,
                )

                yield RegistryEntry(
                    entry_id=entry_id,
                    name=f"FLEURS ({fleurs_config}) — {canonical_split.value}",
                    description=(
                        f"Google FLEURS few-shot evaluation dataset for {bcp47}, "
                        f"{canonical_split.value} split. "
                        "High-quality read speech from diverse speakers."
                    ),
                    language=bcp47,
                    split=canonical_split,
                    domain="read_speech",
                    source_type=SourceType.HUGGINGFACE,
                    hf_repo_id=repo_id,
                    hf_config=fleurs_config,
                    hf_split_name=source_split_name,
                    hf_audio_column=audio_col,
                    hf_transcript_column=transcript_col,
                    hf_streaming_compatible=True,
                    license=self.LICENSE,
                    quality_tier=QualityTier.GOLD,  # FLEURS is carefully curated
                    is_synthetic=False,
                    is_noisy=False,
                    validation_passed=False,
                    stats=stats,
                    provenance=provenance,
                )

    def _detect_fleurs_transcript_col(self, columns: list[str]) -> str:
        """FLEURS uses 'transcription' or 'raw_transcription'."""
        for col in ["transcription", "raw_transcription", "sentence", "text"]:
            if col in columns:
                return col
        raise SchemaIncompatibleError(
            f"No FLEURS transcript column found in: {columns}"
        )
