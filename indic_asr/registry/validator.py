"""
indic_asr/registry/validator.py
=================================
Validates RegistryEntry objects before they are committed to the registry.

Two levels of validation:
1. Structural: Pydantic handles this automatically.
2. Semantic: business-rule checks that Pydantic cannot express.

Semantic checks include:
  - entry_id matches language and split
  - HF repo is accessible (optional, skipped in offline mode)
  - License is not UNKNOWN unless explicitly allowed
  - Stats are internally consistent
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from indic_asr.schema import (
    INDIC_LANGUAGE_CODES,
    LicenseType,
    QualityTier,
    RegistryEntry,
    SplitType,
)

logger = logging.getLogger(__name__)

# entry_id pattern: {source_id}__{lang}__{split}
ENTRY_ID_PATTERN = re.compile(
    r"^[a-zA-Z0-9_\-\.]+__[a-z]{2,3}__(train|validation|test|other)$"
)


class ValidationError:
    def __init__(self, field: str, message: str, severity: str = "error"):
        self.field = field
        self.message = message
        self.severity = severity  # "error" | "warning"

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.field}: {self.message}"


class RegistryValidator:
    """
    Validates RegistryEntry objects semantically.
    Structural validation is handled by Pydantic automatically.
    """

    def __init__(
        self,
        allow_unknown_license: bool = False,
        check_hf_accessibility: bool = False,
    ):
        self.allow_unknown_license = allow_unknown_license
        self.check_hf_accessibility = check_hf_accessibility

    def validate(self, entry: RegistryEntry) -> list[ValidationError]:
        """
        Run all validators. Returns list of ValidationErrors.
        Empty list = valid.
        """
        errors: list[ValidationError] = []

        errors.extend(self._check_entry_id(entry))
        errors.extend(self._check_language(entry))
        errors.extend(self._check_license(entry))
        errors.extend(self._check_stats(entry))
        errors.extend(self._check_hf_fields(entry))
        errors.extend(self._check_provenance(entry))

        if self.check_hf_accessibility:
            errors.extend(self._check_hf_accessible(entry))

        # Log warnings but don't block
        blocking_errors = [e for e in errors if e.severity == "error"]
        warnings = [e for e in errors if e.severity == "warning"]

        for w in warnings:
            logger.warning(str(w))

        return blocking_errors

    def _check_entry_id(self, entry: RegistryEntry) -> list[ValidationError]:
        errors = []

        if not ENTRY_ID_PATTERN.match(entry.entry_id):
            errors.append(ValidationError(
                "entry_id",
                f"Does not match expected pattern "
                f"'{{source_id}}__{{lang}}__{{split}}': {entry.entry_id!r}. "
                f"Pattern: {ENTRY_ID_PATTERN.pattern}"
            ))
            return errors  # Can't do further checks without valid entry_id

        # Check that language in entry_id matches language field
        parts = entry.entry_id.split("__")
        if len(parts) >= 2:
            id_lang = parts[-2]
            if id_lang != entry.language:
                errors.append(ValidationError(
                    "entry_id",
                    f"Language in entry_id ({id_lang!r}) does not match "
                    f"language field ({entry.language!r})"
                ))

        # Check split in entry_id matches split field
        id_split = parts[-1]
        if id_split != entry.split.value:
            errors.append(ValidationError(
                "entry_id",
                f"Split in entry_id ({id_split!r}) does not match "
                f"split field ({entry.split.value!r})"
            ))

        return errors

    def _check_language(self, entry: RegistryEntry) -> list[ValidationError]:
        errors = []

        if entry.language not in INDIC_LANGUAGE_CODES:
            errors.append(ValidationError(
                "language",
                f"Language {entry.language!r} is not in the known Indic language codes. "
                f"If this is intentional, add it to INDIC_LANGUAGE_CODES.",
                severity="warning"
            ))

        return errors

    def _check_license(self, entry: RegistryEntry) -> list[ValidationError]:
        errors = []

        if entry.license == LicenseType.UNKNOWN and not self.allow_unknown_license:
            errors.append(ValidationError(
                "license",
                "License is UNKNOWN. Either set a valid license or run with "
                "allow_unknown_license=True to suppress this error.",
                severity="warning"
            ))

        return errors

    def _check_stats(self, entry: RegistryEntry) -> list[ValidationError]:
        errors = []
        if entry.stats is None:
            return errors

        s = entry.stats
        if s.num_samples < 0:
            errors.append(ValidationError("stats.num_samples", "Must be >= 0"))

        if s.total_duration_seconds < 0:
            errors.append(ValidationError(
                "stats.total_duration_seconds", "Must be >= 0"
            ))

        if s.num_samples > 0 and s.total_duration_seconds == 0:
            errors.append(ValidationError(
                "stats.total_duration_seconds",
                "Is 0 but num_samples > 0. Run validation pipeline to compute.",
                severity="warning"
            ))

        if (
            s.mean_duration_seconds is not None
            and s.num_samples > 0
            and s.total_duration_seconds > 0
        ):
            expected_mean = s.total_duration_seconds / s.num_samples
            if abs(expected_mean - s.mean_duration_seconds) > 0.5:
                errors.append(ValidationError(
                    "stats",
                    f"mean_duration_seconds ({s.mean_duration_seconds:.2f}s) "
                    f"inconsistent with total/num ({expected_mean:.2f}s)",
                    severity="warning"
                ))

        if s.sample_rate_hz is not None and s.sample_rate_hz not in {8000, 16000, 22050, 44100, 48000}:
            errors.append(ValidationError(
                "stats.sample_rate_hz",
                f"Unusual sample rate: {s.sample_rate_hz}Hz. "
                "Most ASR models expect 16000Hz.",
                severity="warning"
            ))

        return errors

    def _check_hf_fields(self, entry: RegistryEntry) -> list[ValidationError]:
        errors = []

        if entry.source_type.value == "huggingface":
            if not entry.hf_repo_id:
                errors.append(ValidationError(
                    "hf_repo_id",
                    "Required for HuggingFace source type"
                ))
            if not entry.hf_audio_column:
                errors.append(ValidationError(
                    "hf_audio_column",
                    "Required for HuggingFace source type"
                ))
            if not entry.hf_transcript_column:
                errors.append(ValidationError(
                    "hf_transcript_column",
                    "Required for HuggingFace source type"
                ))

        return errors

    def _check_provenance(self, entry: RegistryEntry) -> list[ValidationError]:
        errors = []
        p = entry.provenance

        if not p.source_id:
            errors.append(ValidationError("provenance.source_id", "Cannot be empty"))

        if not p.adapter_class:
            errors.append(ValidationError("provenance.adapter_class", "Cannot be empty"))

        if not p.adapter_version:
            errors.append(ValidationError("provenance.adapter_version", "Cannot be empty"))

        if p.hf_commit_sha is None and entry.source_type.value == "huggingface":
            errors.append(ValidationError(
                "provenance.hf_commit_sha",
                "Not recorded. Run with network access for full provenance.",
                severity="warning"
            ))

        return errors

    def _check_hf_accessible(self, entry: RegistryEntry) -> list[ValidationError]:
        """Optional: verify the HF repo actually exists and is accessible."""
        errors = []
        if not entry.hf_repo_id:
            return errors

        try:
            from huggingface_hub import HfApi
            api = HfApi()
            info = api.dataset_info(entry.hf_repo_id)
            if info is None:
                errors.append(ValidationError(
                    "hf_repo_id",
                    f"Repo {entry.hf_repo_id!r} not found on HF Hub"
                ))
        except Exception as e:
            errors.append(ValidationError(
                "hf_repo_id",
                f"Could not verify HF repo accessibility: {e}",
                severity="warning"
            ))

        return errors


def validate_all_entries(registry_dir) -> dict[str, list[str]]:
    """
    Validate all JSON files in registry/entries/.
    Returns dict mapping entry_id → list of error strings.
    """
    import json
    from pathlib import Path

    registry_dir = Path(registry_dir)
    entries_dir = registry_dir / "entries"
    validator = RegistryValidator()
    results = {}

    for json_path in sorted(entries_dir.glob("*.json")):
        try:
            with open(json_path) as f:
                data = json.load(f)
            entry = RegistryEntry(**data)
            errors = validator.validate(entry)
            if errors:
                results[entry.entry_id] = [str(e) for e in errors]
        except Exception as e:
            results[json_path.stem] = [f"Parse error: {e}"]

    return results
