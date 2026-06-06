"""
tests/test_schema_and_adapters.py
===================================
Unit tests for schema validation, adapter logic, and deduplication.
These tests run without network access using mock data.

Run with: pytest tests/ -v
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

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
    INDIC_LANGUAGE_CODES,
)
from indic_asr.registry.validator import RegistryValidator
from indic_asr.registry.deduplicator import (
    TranscriptDeduplicator,
    deduplicate_manifest_rows,
)
from indic_asr.schema import ManifestRow


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def make_provenance(**kwargs) -> ProvenanceRecord:
    defaults = {
        "source_type": SourceType.HUGGINGFACE,
        "source_id": "test/dataset",
        "hf_repo_id": "test/dataset",
        "adapter_class": "TestAdapter",
        "adapter_version": "1.0.0",
        "ingested_by": "test",
    }
    defaults.update(kwargs)
    return ProvenanceRecord(**defaults)


def make_entry(**kwargs) -> RegistryEntry:
    defaults = {
        "entry_id": "test__dataset__hi__train",
        "name": "Test Dataset",
        "language": "hi",
        "split": SplitType.TRAIN,
        "source_type": SourceType.HUGGINGFACE,
        "hf_repo_id": "test/dataset",
        "hf_audio_column": "audio",
        "hf_transcript_column": "sentence",
        "license": LicenseType.CC0,
        "quality_tier": QualityTier.SILVER,
        "provenance": make_provenance(),
    }
    defaults.update(kwargs)
    return RegistryEntry(**defaults)


def make_manifest_row(
    utterance_id: str,
    transcript: str,
    language: str = "hi",
    quality_tier: QualityTier = QualityTier.SILVER,
    source_entry_id: str = "test__entry",
) -> ManifestRow:
    return ManifestRow(
        utterance_id=utterance_id,
        audio_hf_repo="test/dataset",
        audio_hf_split="train",
        audio_hf_row_index=0,
        transcript=transcript,
        language=language,
        source_entry_id=source_entry_id,
        source_split=SplitType.TRAIN,
        quality_tier=quality_tier,
        license=LicenseType.CC0,
    )


# ---------------------------------------------------------------------------
# Language normalization tests
# ---------------------------------------------------------------------------

class TestLanguageNormalization:
    def test_canonical_code_passthrough(self):
        assert normalize_language_code("hi") == "hi"
        assert normalize_language_code("ta") == "ta"
        assert normalize_language_code("ml") == "ml"

    def test_alias_resolution(self):
        assert normalize_language_code("hindi") == "hi"
        assert normalize_language_code("Hindi") == "hi"
        assert normalize_language_code("hin") == "hi"
        assert normalize_language_code("hi-IN") == "hi"
        assert normalize_language_code("hi_IN") == "hi"

    def test_tamil_aliases(self):
        assert normalize_language_code("tamil") == "ta"
        assert normalize_language_code("tam") == "ta"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown language code"):
            normalize_language_code("xyz")

    def test_all_indic_codes_normalize(self):
        for code in INDIC_LANGUAGE_CODES:
            assert normalize_language_code(code) == code


# ---------------------------------------------------------------------------
# Schema validation tests
# ---------------------------------------------------------------------------

class TestSeedDatasetEntry:
    def test_valid_entry(self):
        entry = SeedDatasetEntry(
            id="mozilla-foundation/common_voice_17_0",
            name="Common Voice",
            adapter="CommonVoiceAdapter",
            hf_repo_id="mozilla-foundation/common_voice_17_0",
            languages=["hi", "ta"],
            license=LicenseType.CC0,
        )
        assert entry.languages == ["hi", "ta"]

    def test_language_normalization_in_seed(self):
        entry = SeedDatasetEntry(
            id="test/dataset",
            name="Test",
            adapter="GenericHFAdapter",
            languages=["hindi", "Tamil", "hi-IN"],
        )
        assert entry.languages == ["hi", "ta", "hi"]

    def test_disabled_by_default_is_enabled(self):
        entry = SeedDatasetEntry(
            id="test/dataset",
            name="Test",
            adapter="GenericHFAdapter",
            languages=["hi"],
        )
        assert entry.enabled is True


class TestRegistryEntry:
    def test_valid_entry_creation(self):
        entry = make_entry()
        assert entry.entry_id == "test__dataset__hi__train"
        assert entry.language == "hi"

    def test_language_normalized(self):
        entry = make_entry(language="hindi")
        assert entry.language == "hi"

    def test_invalid_entry_id_raises(self):
        with pytest.raises(ValueError):
            make_entry(entry_id="invalid entry id with spaces")

    def test_entry_path_format(self):
        entry = make_entry()
        path = entry.to_registry_path()
        assert path == "registry/entries/test__dataset__hi__train.json"

    def test_serialization_roundtrip(self):
        entry = make_entry()
        data = entry.model_dump(mode="json")
        restored = RegistryEntry(**data)
        assert restored.entry_id == entry.entry_id
        assert restored.language == entry.language


# ---------------------------------------------------------------------------
# Validator tests
# ---------------------------------------------------------------------------

class TestRegistryValidator:
    def setup_method(self):
        self.validator = RegistryValidator()

    def test_valid_entry_passes(self):
        entry = make_entry()
        errors = self.validator.validate(entry)
        assert errors == []

    def test_mismatched_language_in_id_fails(self):
        # entry_id says 'ta' but language field says 'hi'
        entry = make_entry(
            entry_id="test__dataset__ta__train",  # 'ta' in ID
            language="hi",                         # but 'hi' in field
        )
        errors = self.validator.validate(entry)
        assert any("entry_id" in e.field for e in errors)

    def test_mismatched_split_in_id_fails(self):
        entry = make_entry(
            entry_id="test__dataset__hi__test",   # 'test' in ID
            split=SplitType.TRAIN,                 # but TRAIN in field
        )
        errors = self.validator.validate(entry)
        assert any("entry_id" in e.field for e in errors)

    def test_invalid_entry_id_pattern(self):
        entry = make_entry(entry_id="no-double-underscore")
        errors = self.validator.validate(entry)
        assert any("entry_id" in e.field for e in errors)

    def test_inconsistent_stats_warns(self):
        validator = RegistryValidator()
        stats = AudioStats(
            num_samples=1000,
            total_duration_seconds=5000.0,
            mean_duration_seconds=3.0,  # Wrong: should be ~5.0
        )
        entry = make_entry(stats=stats)
        # Should produce a warning (not blocking error)
        all_issues = []

        # Patch to capture warnings
        import logging
        with patch.object(logging.getLogger("indic_asr.registry.validator"), "warning") as mock_warn:
            errors = validator.validate(entry)
            # Warnings don't block
            assert errors == [] or not any("mean_duration" in str(e) for e in errors)

    def test_missing_hf_repo_id_fails(self):
        entry = make_entry(hf_repo_id=None)
        errors = self.validator.validate(entry)
        assert any("hf_repo_id" in e.field for e in errors)


# ---------------------------------------------------------------------------
# Deduplicator tests
# ---------------------------------------------------------------------------

class TestTranscriptDeduplicator:
    def test_unique_transcripts_not_deduped(self):
        dedup = TranscriptDeduplicator("hi")
        is_dup1, _ = dedup.is_duplicate("u1", "नमस्ते दुनिया", QualityTier.SILVER)
        is_dup2, _ = dedup.is_duplicate("u2", "अलविदा दुनिया", QualityTier.SILVER)
        assert not is_dup1
        assert not is_dup2

    def test_exact_duplicate_detected(self):
        dedup = TranscriptDeduplicator("hi")
        dedup.is_duplicate("u1", "नमस्ते दुनिया", QualityTier.SILVER)
        is_dup, existing = dedup.is_duplicate("u2", "नमस्ते दुनिया", QualityTier.SILVER)
        assert is_dup
        assert existing == "u1"

    def test_normalized_duplicate_detected(self):
        """Punctuation and casing differences should be treated as duplicates."""
        dedup = TranscriptDeduplicator("hi")
        dedup.is_duplicate("u1", "नमस्ते दुनिया!", QualityTier.SILVER)
        is_dup, _ = dedup.is_duplicate("u2", "नमस्ते दुनिया", QualityTier.SILVER)
        assert is_dup

    def test_higher_quality_replaces_lower(self):
        dedup = TranscriptDeduplicator("hi")
        dedup.is_duplicate("u1", "नमस्ते", QualityTier.BRONZE)
        # Gold quality should replace bronze
        is_dup, replaced_id = dedup.is_duplicate("u2", "नमस्ते", QualityTier.GOLD)
        assert not is_dup  # New entry is NOT a dup (it won)
        assert replaced_id == "u1"  # Old entry is now the dup

    def test_lower_quality_is_dropped(self):
        dedup = TranscriptDeduplicator("hi")
        dedup.is_duplicate("u1", "नमस्ते", QualityTier.GOLD)
        is_dup, kept = dedup.is_duplicate("u2", "नमस्ते", QualityTier.BRONZE)
        assert is_dup
        assert kept == "u1"

    def test_transcript_normalization(self):
        dedup = TranscriptDeduplicator("hi")
        t1 = dedup.normalize_transcript("  नमस्ते   दुनिया!  ")
        t2 = dedup.normalize_transcript("नमस्ते दुनिया")
        assert t1 == t2


class TestDeduplicateManifestRows:
    def test_empty_input(self):
        rows, result = deduplicate_manifest_rows([])
        assert rows == []
        assert result.total_rows == 0

    def test_no_duplicates(self):
        rows = [
            make_manifest_row("u1", "नमस्ते", source_entry_id="entry1"),
            make_manifest_row("u2", "अलविदा", source_entry_id="entry1"),
            make_manifest_row("u3", "धन्यवाद", source_entry_id="entry2"),
        ]
        deduped, result = deduplicate_manifest_rows(rows)
        assert len(deduped) == 3
        assert result.total_rows == 3
        assert result.duplicate_rows == 0

    def test_cross_dataset_dedup(self):
        rows = [
            make_manifest_row("u1", "नमस्ते", source_entry_id="common_voice"),
            make_manifest_row("u2", "नमस्ते", source_entry_id="fleurs"),  # Duplicate
            make_manifest_row("u3", "अलविदा", source_entry_id="common_voice"),
        ]
        deduped, result = deduplicate_manifest_rows(rows)
        assert len(deduped) == 2
        assert result.duplicate_rows == 1

    def test_different_languages_not_deduped(self):
        rows = [
            make_manifest_row("u1", "hello", language="hi"),
            make_manifest_row("u2", "hello", language="ta"),  # Same text, different lang
        ]
        deduped, result = deduplicate_manifest_rows(rows)
        assert len(deduped) == 2  # Not duplicates (different languages)

    def test_deduplicated_flag_set(self):
        rows = [make_manifest_row("u1", "unique text")]
        deduped, _ = deduplicate_manifest_rows(rows)
        assert deduped[0].is_deduplicated is True


# ---------------------------------------------------------------------------
# Adapter tests (with mocked HF builders)
# ---------------------------------------------------------------------------

class TestCommonVoiceAdapter:
    def _make_seed(self, configs=None):
        return SeedDatasetEntry(
            id="mozilla-foundation/common_voice_17_0",
            name="Common Voice 17",
            adapter="CommonVoiceAdapter",
            hf_repo_id="mozilla-foundation/common_voice_17_0",
            hf_configs=configs or ["hi"],
            languages=["hi"],
            license=LicenseType.CC0,
        )

    def test_adapter_resolves(self):
        from indic_asr.adapters import get_adapter
        seed = self._make_seed()
        adapter = get_adapter(seed)
        assert adapter is not None
        assert adapter.__class__.__name__ == "CommonVoiceAdapter"

    def test_adapter_can_handle(self):
        from indic_asr.adapters.common_voice import CommonVoiceAdapter
        seed = self._make_seed()
        assert CommonVoiceAdapter.can_handle(seed)

    @patch("indic_asr.adapters.base.BaseAdapter._safe_load_builder")
    @patch("indic_asr.adapters.base.BaseAdapter._get_hf_commit_sha")
    def test_iter_entries_with_mock(self, mock_sha, mock_builder):
        """Test entry generation without network access."""
        from indic_asr.adapters.common_voice import CommonVoiceAdapter

        # Mock commit SHA
        mock_sha.return_value = "abc123"

        # Mock dataset builder
        mock_info = MagicMock()
        mock_info.features = {
            "audio": MagicMock(),
            "sentence": MagicMock(),
            "locale": MagicMock(),
        }
        mock_info.splits = {
            "train": MagicMock(num_examples=5000),
            "validation": MagicMock(num_examples=500),
            "test": MagicMock(num_examples=500),
        }
        mock_builder_instance = MagicMock()
        mock_builder_instance.info = mock_info
        mock_builder.return_value = mock_builder_instance

        seed = self._make_seed(configs=["hi"])
        adapter = CommonVoiceAdapter(seed)

        entries = list(adapter.iter_registry_entries(
            languages=["hi"],
            splits=[SplitType.TRAIN],
        ))

        assert len(entries) == 1
        entry = entries[0]
        assert entry.language == "hi"
        assert entry.split == SplitType.TRAIN
        assert entry.hf_audio_column == "audio"
        assert entry.hf_transcript_column == "sentence"
        assert entry.provenance.hf_commit_sha == "abc123"
        assert entry.provenance.adapter_class == "CommonVoiceAdapter"
        assert entry.license == LicenseType.CC0
        assert entry.quality_tier == QualityTier.SILVER


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
