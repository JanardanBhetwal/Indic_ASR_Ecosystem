"""
indic_asr/registry/builder.py
===============================
Registry builder: runs adapters → writes RegistryEntry JSON files.

Each registry entry is:
  - Written atomically (temp file + rename)
  - Validated before writing
  - Versioned via git (you commit the registry/entries/ directory)
  - Named by entry_id for filesystem clarity

The builder is idempotent: re-running it will overwrite existing entries
if the source has changed (detected via adapter version + schema hash).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import yaml

from indic_asr.adapters import get_adapter
from indic_asr.registry.validator import RegistryValidator
from indic_asr.schema import RegistryEntry, SeedDatasetEntry, SplitType

logger = logging.getLogger(__name__)


class RegistryBuilder:
    """
    Orchestrates the ingest pipeline:
    seed_datasets.yaml → adapters → validated RegistryEntry JSON files
    """

    def __init__(
        self,
        seed_catalogue_path: Path,
        registry_dir: Path,
        overwrite_existing: bool = False,
        dry_run: bool = False,
    ):
        self.seed_path = seed_catalogue_path
        self.registry_dir = registry_dir
        self.overwrite = overwrite_existing
        self.dry_run = dry_run
        self.validator = RegistryValidator()
        self._entries_dir = registry_dir / "entries"
        self._entries_dir.mkdir(parents=True, exist_ok=True)

    def load_seed_catalogue(self) -> list[SeedDatasetEntry]:
        """Load and validate the seed YAML."""
        with open(self.seed_path) as f:
            raw = yaml.safe_load(f)

        datasets = raw.get("datasets", [])
        entries = []
        for item in datasets:
            try:
                entry = SeedDatasetEntry(**item)
                if entry.enabled:
                    entries.append(entry)
                else:
                    logger.debug(f"Skipping disabled entry: {item.get('id')}")
            except Exception as e:
                logger.error(f"Invalid seed entry {item.get('id')}: {e}")

        logger.info(f"Loaded {len(entries)} enabled seed entries")
        return entries

    def build_all(
        self,
        filter_languages: Optional[list[str]] = None,
        filter_splits: Optional[list[SplitType]] = None,
        filter_ids: Optional[list[str]] = None,
    ) -> dict[str, list[str]]:
        """
        Run the full ingest pipeline.
        
        Returns:
            dict with keys 'written', 'skipped', 'failed'
        """
        results = {"written": [], "skipped": [], "failed": []}
        seed_entries = self.load_seed_catalogue()

        for seed in seed_entries:
            if filter_ids and seed.id not in filter_ids:
                continue

            adapter = get_adapter(seed)
            if adapter is None:
                logger.error(f"No adapter for {seed.id}, skipping")
                results["failed"].append(seed.id)
                continue

            logger.info(f"Ingesting {seed.id} with {adapter.__class__.__name__}")

            try:
                for registry_entry in adapter.iter_registry_entries(
                    languages=filter_languages,
                    splits=filter_splits,
                    dry_run=self.dry_run,
                ):
                    outcome = self._write_entry(registry_entry)
                    results[outcome].append(registry_entry.entry_id)

            except Exception as e:
                logger.exception(f"Failed ingesting {seed.id}: {e}")
                results["failed"].append(seed.id)

        self._log_summary(results)
        return results

    def _write_entry(self, entry: RegistryEntry) -> str:
        """
        Validate and write a registry entry. Returns 'written' or 'skipped'.
        """
        out_path = self._entries_dir / f"{entry.entry_id}.json"

        # Skip if already exists and not overwriting
        if out_path.exists() and not self.overwrite:
            existing = self._load_entry(out_path)
            if existing and self._is_unchanged(existing, entry):
                logger.debug(f"Unchanged, skipping: {entry.entry_id}")
                return "skipped"

        # Validate before writing
        errors = self.validator.validate(entry)
        if errors:
            logger.error(f"Validation failed for {entry.entry_id}:")
            for err in errors:
                logger.error(f"  - {err}")
            return "failed"

        if self.dry_run:
            logger.info(f"[DRY RUN] Would write: {out_path}")
            return "written"

        # Atomic write: write to temp, then rename
        self._atomic_write(out_path, entry)
        logger.info(f"Written: {out_path.name}")
        return "written"

    def _atomic_write(self, out_path: Path, entry: RegistryEntry) -> None:
        """Write JSON atomically to avoid partial writes."""
        payload = entry.model_dump(mode="json", indent=2)
        
        # Write to temp file in same directory, then rename
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=out_path.parent,
            prefix=f".{out_path.stem}_",
            suffix=".json.tmp"
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
                f.write("\n")
            os.replace(tmp_path, out_path)
        except Exception:
            os.unlink(tmp_path)
            raise

    def _load_entry(self, path: Path) -> Optional[RegistryEntry]:
        """Load an existing registry entry for comparison."""
        try:
            with open(path) as f:
                data = json.load(f)
            return RegistryEntry(**data)
        except Exception as e:
            logger.warning(f"Could not load existing entry {path}: {e}")
            return None

    def _is_unchanged(self, existing: RegistryEntry, new: RegistryEntry) -> bool:
        """
        Check if the new entry would change anything meaningful.
        Compares adapter version and schema hash as change signals.
        """
        return (
            existing.provenance.adapter_version == new.provenance.adapter_version
            and existing.provenance.schema_hash == new.provenance.schema_hash
            and existing.provenance.hf_commit_sha == new.provenance.hf_commit_sha
        )

    def _log_summary(self, results: dict[str, list[str]]) -> None:
        logger.info("=" * 60)
        logger.info("INGEST SUMMARY")
        logger.info(f"  Written:  {len(results['written'])}")
        logger.info(f"  Skipped:  {len(results['skipped'])}")
        logger.info(f"  Failed:   {len(results['failed'])}")
        if results["failed"]:
            logger.warning("Failed entries:")
            for fid in results["failed"]:
                logger.warning(f"  - {fid}")
        logger.info("=" * 60)
