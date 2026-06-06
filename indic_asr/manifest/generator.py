"""
indic_asr/manifest/generator.py
==================================
Generates model-ready Parquet manifests from registry entries.

A manifest is the public-facing product: a flat Parquet file
that any ASR training pipeline can consume directly.

Each manifest is:
  - One language, one split (e.g., hi_train.parquet)
  - Fully deduplicated across all source datasets
  - Quality-filtered (configurable)
  - License-filtered (configurable)
  - Reproducible: same registry → same manifest

Manifests do NOT contain audio bytes. They contain:
  - Streaming pointers to HF Hub audio
  - Normalized transcripts
  - Full provenance metadata per row

Usage in training:
  from datasets import load_dataset
  ds = load_dataset("parquet", data_files={"train": "hi_train.parquet"})
  # Then stream audio from HF Hub using the pointer fields
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from indic_asr.registry.deduplicator import deduplicate_manifest_rows
from indic_asr.schema import (
    LicenseType,
    ManifestMetadata,
    ManifestRow,
    QualityTier,
    RegistryEntry,
    SplitType,
)

logger = logging.getLogger(__name__)

GENERATOR_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Transcript normalization
# ---------------------------------------------------------------------------

# Unicode block ranges for Indic scripts
INDIC_SCRIPT_RANGES = [
    (0x0900, 0x097F),  # Devanagari (Hindi, Marathi, Sanskrit, Nepali)
    (0x0980, 0x09FF),  # Bengali / Assamese
    (0x0A00, 0x0A7F),  # Gurmukhi (Punjabi)
    (0x0A80, 0x0AFF),  # Gujarati
    (0x0B00, 0x0B7F),  # Odia
    (0x0B80, 0x0BFF),  # Tamil
    (0x0C00, 0x0C7F),  # Telugu
    (0x0C80, 0x0CFF),  # Kannada
    (0x0D00, 0x0D7F),  # Malayalam
    (0x0E00, 0x0E7F),  # Thai (for multilingual future expansion)
    (0x0600, 0x06FF),  # Arabic (Urdu uses Nastaliq)
]


def normalize_transcript(text: str, language: str) -> str:
    """
    Normalize a transcript for acoustic model training.
    
    Conservative normalization that:
    - Applies Unicode NFC
    - Lowercases Latin characters only (preserves case in scripts that use it)
    - Removes leading/trailing whitespace
    - Collapses internal whitespace
    - Strips common punctuation (keeps Indic-specific punctuation if meaningful)
    
    Intentionally does NOT:
    - Remove numerals (many Indic ASR models handle numerals)
    - Convert numerals to words (task-specific, do it in your training pipeline)
    - Strip diacritics (critical for meaning in Indic scripts)
    """
    # Unicode normalization (NFC preserves Indic character compositions)
    text = unicodedata.normalize("NFC", text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    # Collapse internal whitespace
    text = re.sub(r"\s+", " ", text)
    
    # Remove common ASCII punctuation that doesn't affect pronunciation
    # Keep: spaces, Indic digits, Arabic digits, Indic characters
    # Remove: ! . , ? ; : " ' - ( ) [ ] { } / \ | @ # $ % ^ & * + = < > `
    ascii_punct_to_remove = '!"\'(),./:;<=>?@[\\]^_`{|}~'
    text = text.translate(str.maketrans("", "", ascii_punct_to_remove))
    
    # Collapse whitespace again after punct removal
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


# ---------------------------------------------------------------------------
# Manifest filters
# ---------------------------------------------------------------------------


class ManifestFilters:
    """Configurable filters applied during manifest generation."""

    def __init__(
        self,
        min_duration_s: float = 0.5,
        max_duration_s: float = 30.0,
        min_transcript_chars: int = 2,
        max_transcript_chars: int = 500,
        allowed_licenses: Optional[list[LicenseType]] = None,
        allowed_quality_tiers: Optional[list[QualityTier]] = None,
        exclude_synthetic: bool = False,
        exclude_noisy: bool = False,
    ):
        self.min_duration_s = min_duration_s
        self.max_duration_s = max_duration_s
        self.min_transcript_chars = min_transcript_chars
        self.max_transcript_chars = max_transcript_chars
        self.allowed_licenses = allowed_licenses
        self.allowed_quality_tiers = allowed_quality_tiers
        self.exclude_synthetic = exclude_synthetic
        self.exclude_noisy = exclude_noisy

    def passes(self, row: ManifestRow, entry: RegistryEntry) -> tuple[bool, str]:
        """
        Check if a row passes all filters.
        Returns (passes, reason_if_failed).
        """
        if self.exclude_synthetic and entry.is_synthetic:
            return False, "synthetic"
        
        if self.exclude_noisy and entry.is_noisy:
            return False, "noisy"
        
        if self.allowed_licenses and entry.license not in self.allowed_licenses:
            return False, f"license_excluded ({entry.license.value})"
        
        if self.allowed_quality_tiers and entry.quality_tier not in self.allowed_quality_tiers:
            return False, f"quality_tier_excluded ({entry.quality_tier.value})"
        
        if row.duration_seconds is not None:
            if row.duration_seconds < self.min_duration_s:
                return False, f"too_short ({row.duration_seconds:.2f}s)"
            if row.duration_seconds > self.max_duration_s:
                return False, f"too_long ({row.duration_seconds:.2f}s)"
        
        transcript_len = len(row.transcript.strip())
        if transcript_len < self.min_transcript_chars:
            return False, f"transcript_too_short ({transcript_len} chars)"
        if transcript_len > self.max_transcript_chars:
            return False, f"transcript_too_long ({transcript_len} chars)"
        
        return True, ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_duration_s": self.min_duration_s,
            "max_duration_s": self.max_duration_s,
            "min_transcript_chars": self.min_transcript_chars,
            "max_transcript_chars": self.max_transcript_chars,
            "allowed_licenses": (
                [l.value for l in self.allowed_licenses]
                if self.allowed_licenses else None
            ),
            "allowed_quality_tiers": (
                [t.value for t in self.allowed_quality_tiers]
                if self.allowed_quality_tiers else None
            ),
            "exclude_synthetic": self.exclude_synthetic,
            "exclude_noisy": self.exclude_noisy,
        }


# ---------------------------------------------------------------------------
# Manifest generator
# ---------------------------------------------------------------------------


class ManifestGenerator:
    """
    Builds Parquet manifests from a set of RegistryEntry objects.
    
    The generator is stateless: given the same registry entries and filters,
    it produces the same manifest every time.
    """

    def __init__(
        self,
        registry_dir: Path,
        output_dir: Path,
        filters: Optional[ManifestFilters] = None,
        deduplicate: bool = True,
    ):
        self.registry_dir = Path(registry_dir)
        self.output_dir = Path(output_dir)
        self.filters = filters or ManifestFilters()
        self.deduplicate = deduplicate
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_all_entries(self) -> list[RegistryEntry]:
        """Load all validated registry entries."""
        entries = []
        entries_dir = self.registry_dir / "entries"
        
        for json_path in sorted(entries_dir.glob("*.json")):
            try:
                with open(json_path) as f:
                    data = json.load(f)
                entry = RegistryEntry(**data)
                entries.append(entry)
            except Exception as e:
                logger.error(f"Could not load registry entry {json_path}: {e}")
        
        logger.info(f"Loaded {len(entries)} registry entries")
        return entries

    def build_manifest(
        self,
        language: str,
        split: SplitType,
        entries: Optional[list[RegistryEntry]] = None,
        streaming: bool = True,
    ) -> Path:
        """
        Build a manifest for one language+split combination.
        
        Args:
            language: BCP-47 language code
            split: Train/validation/test
            entries: Pre-loaded registry entries (loaded from disk if None)
            streaming: Use HF streaming to avoid downloading full datasets
        
        Returns:
            Path to the written Parquet file
        """
        if entries is None:
            entries = self.load_all_entries()

        # Filter to relevant entries
        relevant = [
            e for e in entries
            if e.language == language and e.split == split
        ]

        if not relevant:
            logger.warning(f"No registry entries for {language}/{split.value}")
            return None

        logger.info(
            f"Building manifest for {language}/{split.value} "
            f"from {len(relevant)} registry entries"
        )

        # Generate rows from each entry
        all_rows: list[ManifestRow] = []
        filter_stats: dict[str, int] = {}

        for entry in relevant:
            rows, filtered = self._rows_from_entry(entry, split)
            all_rows.extend(rows)
            for reason, count in filtered.items():
                filter_stats[reason] = filter_stats.get(reason, 0) + count

            logger.info(
                f"  {entry.entry_id}: {len(rows)} rows "
                f"({sum(filtered.values())} filtered)"
            )

        if not all_rows:
            logger.warning(f"No rows remaining after filtering for {language}/{split.value}")
            return None

        logger.info(f"Total rows before dedup: {len(all_rows)}")

        # Deduplication
        if self.deduplicate:
            all_rows, dedup_result = deduplicate_manifest_rows(all_rows)
            logger.info(
                f"After dedup: {len(all_rows)} rows "
                f"({dedup_result.duplicate_rows} removed)"
            )

        # Write Parquet
        out_path = self.output_dir / f"{language}_{split.value}.parquet"
        self._write_parquet(all_rows, out_path)

        # Write sidecar metadata
        total_dur = sum(
            r.duration_seconds for r in all_rows if r.duration_seconds is not None
        )
        metadata = ManifestMetadata(
            manifest_id=f"{language}_{split.value}",
            language=language,
            split=split,
            generated_at=datetime.now(timezone.utc),
            generator_version=GENERATOR_VERSION,
            num_rows=len(all_rows),
            total_duration_seconds=total_dur,
            source_entry_ids=[e.entry_id for e in relevant],
            filters_applied=self.filters.to_dict(),
        )
        meta_path = self.output_dir / f"{language}_{split.value}_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2, default=str)

        logger.info(f"Manifest written: {out_path} ({len(all_rows)} rows)")
        return out_path

    def build_all_manifests(
        self,
        languages: Optional[list[str]] = None,
        splits: Optional[list[SplitType]] = None,
    ) -> dict[str, Path]:
        """Build manifests for all language+split combinations in the registry."""
        entries = self.load_all_entries()
        
        # Collect all unique (language, split) pairs
        pairs: set[tuple[str, SplitType]] = {
            (e.language, e.split) for e in entries
        }

        if languages:
            pairs = {(lang, split) for lang, split in pairs if lang in languages}
        if splits:
            pairs = {(lang, split) for lang, split in pairs if split in splits}

        results = {}
        for language, split in sorted(pairs):
            try:
                path = self.build_manifest(language, split, entries)
                if path:
                    results[f"{language}_{split.value}"] = path
            except Exception as e:
                logger.exception(f"Failed to build manifest for {language}/{split.value}: {e}")

        logger.info(f"Built {len(results)} manifests")
        return results

    def _rows_from_entry(
        self,
        entry: RegistryEntry,
        split: SplitType,
    ) -> tuple[list[ManifestRow], dict[str, int]]:
        """
        Generate ManifestRows from a registry entry.
        
        This method inspects the source dataset metadata WITHOUT downloading audio.
        Audio pointers (row indices) are used for reproducible streaming access.
        
        Returns: (rows, filter_reason_counts)
        """
        rows = []
        filter_counts: dict[str, int] = {}

        if not entry.hf_repo_id:
            logger.warning(f"Entry {entry.entry_id} has no HF repo ID, skipping")
            return rows, filter_counts

        try:
            from datasets import load_dataset

            # Use streaming to avoid downloading the full dataset
            # We only need metadata (row count) at this point if stats are available
            num_samples = entry.stats.num_samples if entry.stats else None

            if num_samples is not None and num_samples > 0:
                # Generate pointer-only rows from stats (fast path)
                rows = self._generate_pointer_rows(entry, num_samples)
            else:
                # Need to actually count rows (slower, but still streaming)
                rows = self._generate_rows_with_inspection(entry)

        except Exception as e:
            logger.error(f"Error generating rows for {entry.entry_id}: {e}")

        # Apply filters
        filtered_rows = []
        for row in rows:
            passes, reason = self.filters.passes(row, entry)
            if passes:
                filtered_rows.append(row)
            else:
                filter_counts[reason] = filter_counts.get(reason, 0) + 1

        return filtered_rows, filter_counts

    def _generate_pointer_rows(
        self,
        entry: RegistryEntry,
        num_samples: int,
    ) -> list[ManifestRow]:
        """
        Fast path: generate pointer-only rows when we know the row count.
        
        These rows do not have transcripts yet — transcripts are fetched
        lazily during training via the HF streaming interface.
        
        For manifest purposes, we create placeholder rows with pointers.
        The actual transcript is fetched during training.
        
        In practice, you would want to pre-fetch transcripts for deduplication.
        See _generate_rows_with_inspection for the full version.
        """
        rows = []
        for i in range(num_samples):
            row = ManifestRow(
                utterance_id=f"{entry.entry_id}__{i:08d}",
                audio_hf_repo=entry.hf_repo_id,
                audio_hf_config=entry.hf_config,
                audio_hf_split=entry.hf_split_name or entry.split.value,
                audio_hf_row_index=i,
                transcript="",  # Fetched lazily during training
                language=entry.language,
                source_entry_id=entry.entry_id,
                source_split=entry.split,
                quality_tier=entry.quality_tier,
                license=entry.license,
                is_deduplicated=False,
            )
            rows.append(row)
        return rows

    def _generate_rows_with_inspection(
        self,
        entry: RegistryEntry,
        max_rows: Optional[int] = None,
    ) -> list[ManifestRow]:
        """
        Full path: stream through dataset to get transcripts for deduplication.
        
        This is the production-quality path that enables transcript-level
        deduplication across datasets.
        """
        from datasets import load_dataset

        rows = []
        
        try:
            dataset = load_dataset(
                entry.hf_repo_id,
                entry.hf_config,
                split=entry.hf_split_name or entry.split.value,
                streaming=True,
                trust_remote_code=False,
            )

            for i, sample in enumerate(dataset):
                if max_rows and i >= max_rows:
                    break

                transcript = sample.get(entry.hf_transcript_column, "")
                if not isinstance(transcript, str):
                    transcript = str(transcript) if transcript is not None else ""

                normalized = normalize_transcript(transcript, entry.language)

                row = ManifestRow(
                    utterance_id=f"{entry.entry_id}__{i:08d}",
                    audio_hf_repo=entry.hf_repo_id,
                    audio_hf_config=entry.hf_config,
                    audio_hf_split=entry.hf_split_name or entry.split.value,
                    audio_hf_row_index=i,
                    transcript=transcript,
                    transcript_normalized=normalized,
                    language=entry.language,
                    source_entry_id=entry.entry_id,
                    source_split=entry.split,
                    quality_tier=entry.quality_tier,
                    license=entry.license,
                    is_deduplicated=False,
                )
                rows.append(row)

        except Exception as e:
            logger.error(
                f"Error streaming {entry.hf_repo_id}/{entry.hf_config}/{entry.hf_split_name}: {e}"
            )

        return rows

    def _write_parquet(self, rows: list[ManifestRow], out_path: Path) -> None:
        """Write ManifestRows to Parquet using pyarrow."""
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            # Convert to dicts
            records = [row.model_dump(mode="json") for row in rows]

            # Infer schema from first record
            table = pa.Table.from_pylist(records)

            pq.write_table(
                table,
                out_path,
                compression="snappy",
                # Store schema version in file metadata
                custom_metadata={
                    "indic_asr_schema_version": "1.0",
                    "generator_version": GENERATOR_VERSION,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            logger.info(f"Parquet written: {out_path} ({len(rows)} rows)")

        except ImportError:
            # Fallback: write as JSONL if pyarrow not available
            logger.warning("pyarrow not available, falling back to JSONL")
            jsonl_path = out_path.with_suffix(".jsonl")
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row.model_dump(mode="json"), ensure_ascii=False) + "\n")
            logger.info(f"JSONL written: {jsonl_path} ({len(rows)} rows)")
