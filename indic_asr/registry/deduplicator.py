"""
indic_asr/registry/deduplicator.py
=====================================
Cross-dataset deduplication for the manifest generation pipeline.

Problem:
  Common Voice Hindi v11, v12, v13, v14, v15, v16, v17 all share utterances.
  FLEURS Hindi is derived from Common Voice.
  Shrutilipi may overlap with other broadcast sources.
  
  Naive aggregation double-counts utterances, inflates dataset statistics,
  and causes overfitting in models trained on concatenated data.

Strategy:
  1. Transcript-level dedup (fast, language-independent)
     - Normalize transcript → compute SHA256
     - Flag duplicates across datasets
  
  2. Audio fingerprint dedup (accurate, expensive)
     - Hash first N bytes of audio waveform (not filename, not metadata)
     - Only run on suspected duplicates from transcript-level pass
  
  3. Dataset priority ordering
     - When duplicate found, keep the entry from the higher-priority dataset
     - Priority: GOLD > SILVER > BRONZE, then by dataset quality score

Design note:
  We never mutate registry entries. Deduplication is applied at manifest
  generation time, not at registry write time. The registry is the truth;
  the manifest is the filtered/deduplicated view.
"""

from __future__ import annotations

import hashlib
import logging
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

from indic_asr.schema import ManifestRow, QualityTier, RegistryEntry

logger = logging.getLogger(__name__)


# Priority ordering for deduplication (higher index = higher priority)
QUALITY_PRIORITY: dict[QualityTier, int] = {
    QualityTier.UNKNOWN: 0,
    QualityTier.BRONZE: 1,
    QualityTier.SILVER: 2,
    QualityTier.GOLD: 3,
}


@dataclass
class DeduplicationResult:
    total_rows: int = 0
    unique_rows: int = 0
    duplicate_rows: int = 0
    dedup_groups: int = 0
    kept_by_entry: dict[str, int] = field(default_factory=dict)
    removed_by_entry: dict[str, int] = field(default_factory=dict)


class TranscriptDeduplicator:
    """
    Fast first-pass deduplication using normalized transcripts.
    
    This catches the most common case: the same utterance text
    appearing in multiple datasets (e.g., Common Voice utterances
    reused across versions or downstream datasets like FLEURS).
    
    Note: Different recordings of the same text are NOT duplicates.
    Use audio fingerprinting to distinguish them.
    """

    def __init__(self, language: str):
        self.language = language
        self._seen: dict[str, tuple[str, QualityTier]] = {}
        # fingerprint → (utterance_id, quality_tier)

    def normalize_transcript(self, text: str) -> str:
        """
        Normalize transcript for comparison.
        
        Steps:
        1. Unicode NFC normalization
        2. Lowercase
        3. Remove punctuation (keep Indic characters, digits, spaces)
        4. Collapse whitespace
        """
        # NFC normalization (important for Devanagari, Bengali, etc.)
        text = unicodedata.normalize("NFC", text)
        text = text.lower()
        
        # Remove ASCII punctuation but keep Unicode letters/digits/spaces
        # This preserves Devanagari, Tamil, etc. characters
        cleaned_chars = []
        for ch in text:
            cat = unicodedata.category(ch)
            if cat.startswith("L") or cat.startswith("N") or cat == "Zs":
                cleaned_chars.append(ch)
            elif ch == " ":
                cleaned_chars.append(ch)
        
        text = "".join(cleaned_chars)
        text = " ".join(text.split())  # Normalize whitespace
        return text.strip()

    def transcript_fingerprint(self, text: str) -> str:
        """SHA256 of normalized transcript."""
        normalized = self.normalize_transcript(text)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def is_duplicate(
        self,
        utterance_id: str,
        transcript: str,
        quality_tier: QualityTier,
    ) -> tuple[bool, Optional[str]]:
        """
        Check if this utterance is a duplicate.
        
        Returns:
            (is_dup, kept_utterance_id)
            If not a duplicate: (False, None)
            If duplicate and this one should be dropped: (True, kept_id)
            If duplicate and this one should replace existing: (False, None)
              (also removes the previous one from seen)
        """
        fp = self.transcript_fingerprint(transcript)
        
        if fp not in self._seen:
            self._seen[fp] = (utterance_id, quality_tier)
            return False, None
        
        existing_id, existing_tier = self._seen[fp]
        
        # Higher quality tier wins
        if QUALITY_PRIORITY[quality_tier] > QUALITY_PRIORITY[existing_tier]:
            # New entry is better; replace the existing one
            logger.debug(
                f"Dedup: replacing {existing_id} with {utterance_id} "
                f"(quality: {existing_tier.value} → {quality_tier.value})"
            )
            self._seen[fp] = (utterance_id, quality_tier)
            # Signal: this is not a dup, but the previous entry should be marked as dup
            return False, existing_id  # existing_id is now the duplicate
        
        # Existing entry is equal or better; drop the new one
        return True, existing_id


class AudioFingerprintDeduplicator:
    """
    High-accuracy deduplication using audio content hashes.
    
    This is expensive (requires loading audio data), so it is only run
    when transcript-level dedup flags potential duplicates.
    
    The fingerprint is: SHA256(first_2_seconds_of_raw_pcm_bytes)
    
    This is robust to:
    - Different audio formats (MP3 vs WAV vs FLAC of same recording)
    - Different metadata (different filenames, different dataset fields)
    
    It is NOT robust to:
    - Re-encodings (lossy re-compression changes bytes)
    - Different recordings of the same text
    
    For our purposes (detecting the exact same recording reused across datasets),
    this is sufficient.
    """

    FINGERPRINT_DURATION_S: float = 2.0  # Use first 2 seconds
    SAMPLE_RATE: int = 16000

    def __init__(self):
        self._seen: dict[str, str] = {}  # fingerprint → utterance_id

    @staticmethod
    def compute_fingerprint(audio_array, sample_rate: int) -> str:
        """
        Compute a fingerprint from a numpy audio array.
        
        Args:
            audio_array: numpy array of audio samples (float32, -1 to 1)
            sample_rate: sample rate in Hz
        
        Returns:
            hex string fingerprint
        """
        import numpy as np
        
        # Resample to canonical 16kHz if needed
        target_samples = int(AudioFingerprintDeduplicator.FINGERPRINT_DURATION_S * 16000)
        
        if sample_rate != 16000:
            # Simple linear interpolation for resampling
            # For production, use librosa.resample or scipy.signal.resample
            import math
            factor = 16000 / sample_rate
            target_len = int(len(audio_array) * factor)
            indices = np.linspace(0, len(audio_array) - 1, target_len)
            audio_array = np.interp(indices, np.arange(len(audio_array)), audio_array)
        
        # Take first N samples
        fingerprint_samples = audio_array[:target_samples]
        
        # Quantize to 16-bit integers for stable hashing
        quantized = (fingerprint_samples * 32767).astype(np.int16)
        
        return hashlib.sha256(quantized.tobytes()).hexdigest()

    def is_duplicate(
        self,
        utterance_id: str,
        audio_array,
        sample_rate: int,
    ) -> tuple[bool, Optional[str]]:
        fp = self.compute_fingerprint(audio_array, sample_rate)
        if fp in self._seen:
            return True, self._seen[fp]
        self._seen[fp] = utterance_id
        return False, None


def deduplicate_manifest_rows(
    rows: list[ManifestRow],
    strategy: str = "transcript",  # "transcript" | "audio" | "both"
) -> tuple[list[ManifestRow], DeduplicationResult]:
    """
    Deduplicate a list of ManifestRows.
    
    Args:
        rows: Input rows (may contain duplicates across datasets)
        strategy: Deduplication strategy to use
    
    Returns:
        (deduplicated_rows, DeduplicationResult stats)
    """
    if not rows:
        return [], DeduplicationResult()

    result = DeduplicationResult(total_rows=len(rows))

    # Group by language first (no cross-language dedup)
    lang_groups: dict[str, list[ManifestRow]] = defaultdict(list)
    for row in rows:
        lang_groups[row.language].append(row)

    deduplicated: list[ManifestRow] = []

    for lang, lang_rows in lang_groups.items():
        deduplicator = TranscriptDeduplicator(lang)
        
        kept: list[ManifestRow] = []
        dropped_ids: set[str] = set()

        for row in lang_rows:
            is_dup, existing_id = deduplicator.is_duplicate(
                row.utterance_id,
                row.transcript,
                row.quality_tier,
            )

            if is_dup:
                dropped_ids.add(row.utterance_id)
                result.removed_by_entry[row.source_entry_id] = (
                    result.removed_by_entry.get(row.source_entry_id, 0) + 1
                )
            elif existing_id is not None:
                # This row replaced a lower-quality duplicate
                # Remove the old one from kept list
                kept = [r for r in kept if r.utterance_id != existing_id]
                dropped_ids.add(existing_id)
                kept.append(row)
                result.kept_by_entry[row.source_entry_id] = (
                    result.kept_by_entry.get(row.source_entry_id, 0) + 1
                )
            else:
                kept.append(row)
                result.kept_by_entry[row.source_entry_id] = (
                    result.kept_by_entry.get(row.source_entry_id, 0) + 1
                )

        # Mark kept rows as deduplicated
        kept_marked = [
            ManifestRow(**{**row.model_dump(), "is_deduplicated": True})
            for row in kept
        ]
        deduplicated.extend(kept_marked)
        result.duplicate_rows += len(dropped_ids)

    result.unique_rows = len(deduplicated)
    result.dedup_groups = result.total_rows - result.unique_rows

    logger.info(
        f"Deduplication complete: {result.total_rows} → {result.unique_rows} rows "
        f"({result.duplicate_rows} duplicates removed)"
    )

    return deduplicated, result
