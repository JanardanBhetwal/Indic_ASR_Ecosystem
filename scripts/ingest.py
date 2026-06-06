"""
scripts/ingest.py
==================
Main ingest pipeline: seed catalogue → registry entries.

This is the primary script to run when adding new datasets
or refreshing existing ones.

Usage examples:
    # Ingest all enabled datasets
    python scripts/ingest.py

    # Ingest only specific datasets
    python scripts/ingest.py --ids mozilla-foundation/common_voice_17_0 google/fleurs

    # Ingest only Hindi datasets
    python scripts/ingest.py --langs hi

    # Dry run (no writes)
    python scripts/ingest.py --dry-run

    # Force refresh (overwrite existing entries)
    python scripts/ingest.py --overwrite

In Google Colab:
    !python /content/indic_asr_ecosystem/scripts/ingest.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path for Colab compatibility
sys.path.insert(0, str(Path(__file__).parent.parent))

from indic_asr.registry.builder import RegistryBuilder
from indic_asr.schema import SplitType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Ingest Indic ASR datasets from seed catalogue into registry"
    )
    parser.add_argument(
        "--seed",
        default=None,
        help="Path to seed_datasets.yaml (auto-detected if not specified)"
    )
    parser.add_argument(
        "--registry",
        default=None,
        help="Path to registry directory (auto-detected if not specified)"
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        default=None,
        help="Specific seed entry IDs to ingest (default: all enabled)"
    )
    parser.add_argument(
        "--langs",
        nargs="+",
        default=None,
        help="Filter to specific languages (BCP-47 codes)"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "validation", "test"],
        default=None,
        help="Filter to specific splits"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing registry entries"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate ingest without writing files"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging"
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Resolve paths
    project_root = Path(__file__).parent.parent
    seed_path = Path(args.seed) if args.seed else project_root / "catalogue" / "seed_datasets.yaml"
    registry_dir = Path(args.registry) if args.registry else project_root / "registry"

    if not seed_path.exists():
        logger.error(f"Seed catalogue not found: {seed_path}")
        logger.error("Create it first or specify --seed path")
        sys.exit(1)

    # Parse split filters
    split_filter = None
    if args.splits:
        split_map = {
            "train": SplitType.TRAIN,
            "validation": SplitType.VALIDATION,
            "test": SplitType.TEST,
        }
        split_filter = [split_map[s] for s in args.splits]

    logger.info("=" * 60)
    logger.info("INDIC ASR INGEST PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Seed catalogue: {seed_path}")
    logger.info(f"Registry directory: {registry_dir}")
    logger.info(f"Languages filter: {args.langs or 'all'}")
    logger.info(f"Splits filter: {args.splits or 'all'}")
    logger.info(f"IDs filter: {args.ids or 'all enabled'}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info(f"Overwrite: {args.overwrite}")
    logger.info("=" * 60)

    builder = RegistryBuilder(
        seed_catalogue_path=seed_path,
        registry_dir=registry_dir,
        overwrite_existing=args.overwrite,
        dry_run=args.dry_run,
    )

    results = builder.build_all(
        filter_languages=args.langs,
        filter_splits=split_filter,
        filter_ids=args.ids,
    )

    # Exit with error code if any failures
    if results["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
