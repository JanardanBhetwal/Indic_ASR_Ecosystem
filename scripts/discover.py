"""
scripts/discover.py
====================
Targeted discovery of Indic ASR datasets on HuggingFace Hub.

This replaces the naive list_datasets(full=True) approach that
scans 16,000+ datasets. Instead, we use:

  1. Keyword search via HF Hub API (fast: ~10-20 targeted searches)
  2. Known organization filtering (AI4Bharat, Mozilla, Google, etc.)
  3. Language tag filtering
  4. Community-maintained curated lists

The output is a YAML file of candidate datasets for human review,
NOT automatically added to the registry. A human decides which
candidates to add to seed_datasets.yaml.

Usage:
    python scripts/discover.py --output discovered_candidates.yaml
    python scripts/discover.py --org ai4bharat
    python scripts/discover.py --langs hi,bn,ta
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# Trusted organizations known to publish Indic ASR datasets
TRUSTED_ORGS = [
    "ai4bharat",
    "mozilla-foundation",
    "google",
    "openslr",
    "iitm-lab",
    "IISc-MILE",
    "SMC",          # Swathanthra Malayalam Computing
    "ulca-bhashadaan",
    "speechcolab",
    "Microsoft",
    "espnet",
    "NPTEL-Open-Source",
]

# Search keywords likely to match Indic ASR datasets
SEARCH_KEYWORDS = [
    "indic asr",
    "hindi speech",
    "bengali speech",
    "tamil speech",
    "telugu speech",
    "kannada speech",
    "malayalam speech",
    "marathi speech",
    "gujarati speech",
    "punjabi speech",
    "urdu speech",
    "assamese speech",
    "odia speech",
    "nepali speech",
    "indian language asr",
    "bharat speech",
    "indic language recognition",
]

# Language tags used on HF Hub for Indic languages
INDIC_LANGUAGE_TAGS = [
    "hi", "bn", "as", "gu", "mr", "pa", "ur", "or",
    "ta", "te", "kn", "ml", "ne", "sa", "mai"
]

# Task tags relevant to ASR
ASR_TASK_TAGS = [
    "automatic-speech-recognition",
    "audio",
    "speech-recognition",
]


def search_by_keyword(keyword: str, limit: int = 50) -> list[dict]:
    """Search HF Hub by keyword. Returns list of dataset metadata dicts."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        
        results = list(api.list_datasets(
            search=keyword,
            task_categories=["automatic-speech-recognition"],
            limit=limit,
            full=False,  # Don't fetch full metadata — too slow
        ))
        
        return [
            {
                "id": ds.id,
                "downloads": getattr(ds, "downloads", 0),
                "likes": getattr(ds, "likes", 0),
                "tags": getattr(ds, "tags", []),
                "created_at": str(getattr(ds, "created_at", "")),
                "found_via": f"keyword:{keyword}",
            }
            for ds in results
        ]
    except Exception as e:
        logger.error(f"Search failed for keyword {keyword!r}: {e}")
        return []


def search_by_org(org: str, limit: int = 100) -> list[dict]:
    """Get all datasets from a trusted organization."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        
        results = list(api.list_datasets(
            author=org,
            task_categories=["automatic-speech-recognition"],
            limit=limit,
            full=False,
        ))
        
        return [
            {
                "id": ds.id,
                "downloads": getattr(ds, "downloads", 0),
                "likes": getattr(ds, "likes", 0),
                "tags": getattr(ds, "tags", []),
                "created_at": str(getattr(ds, "created_at", "")),
                "found_via": f"org:{org}",
            }
            for ds in results
        ]
    except Exception as e:
        logger.error(f"Org search failed for {org!r}: {e}")
        return []


def search_by_language_tag(lang_code: str, limit: int = 50) -> list[dict]:
    """Search for datasets tagged with a specific Indic language."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        
        results = list(api.list_datasets(
            language=lang_code,
            task_categories=["automatic-speech-recognition"],
            limit=limit,
            full=False,
        ))
        
        return [
            {
                "id": ds.id,
                "downloads": getattr(ds, "downloads", 0),
                "likes": getattr(ds, "likes", 0),
                "tags": getattr(ds, "tags", []),
                "created_at": str(getattr(ds, "created_at", "")),
                "found_via": f"language_tag:{lang_code}",
            }
            for ds in results
        ]
    except Exception as e:
        logger.error(f"Language tag search failed for {lang_code!r}: {e}")
        return []


def discover_all(
    orgs: list[str] = None,
    keywords: list[str] = None,
    langs: list[str] = None,
) -> list[dict]:
    """
    Run all discovery strategies and return deduplicated candidates.
    
    This is the recommended entry point. It runs ~50 targeted searches
    instead of scanning all 16,000+ HF datasets.
    """
    all_results: dict[str, dict] = {}  # id → metadata (dedup by id)

    orgs = orgs or TRUSTED_ORGS
    keywords = keywords or SEARCH_KEYWORDS
    langs = langs or INDIC_LANGUAGE_TAGS

    # 1. Organization search (most reliable signal)
    logger.info(f"Searching {len(orgs)} organizations...")
    for org in orgs:
        logger.info(f"  Searching org: {org}")
        for result in search_by_org(org):
            if result["id"] not in all_results:
                all_results[result["id"]] = result

    # 2. Keyword search
    logger.info(f"Running {len(keywords)} keyword searches...")
    for keyword in keywords:
        logger.info(f"  Keyword: {keyword!r}")
        for result in search_by_keyword(keyword):
            if result["id"] not in all_results:
                all_results[result["id"]] = result

    # 3. Language tag search
    logger.info(f"Searching {len(langs)} language tags...")
    for lang in langs:
        for result in search_by_language_tag(lang):
            if result["id"] not in all_results:
                all_results[result["id"]] = result

    candidates = list(all_results.values())
    logger.info(f"Discovered {len(candidates)} unique candidate datasets")
    return candidates


def filter_candidates(candidates: list[dict]) -> list[dict]:
    """
    Apply heuristic filtering to reduce noise.
    
    Filters out:
    - Datasets with no downloads (likely empty/test repos)
    - Non-audio datasets that matched our tags by accident
    - Known duplicates (same dataset different versions: keep latest)
    """
    filtered = []
    for c in candidates:
        # Require at least some community interest
        if c.get("downloads", 0) < 10 and c.get("likes", 0) < 2:
            logger.debug(f"Filtered (no activity): {c['id']}")
            continue
        
        # Check tags suggest audio content
        tags = c.get("tags", [])
        tag_str = " ".join(str(t).lower() for t in tags)
        has_audio_signal = any(
            signal in tag_str
            for signal in ["audio", "speech", "asr", "automatic-speech"]
        )
        if tags and not has_audio_signal:
            logger.debug(f"Filtered (no audio tags): {c['id']}")
            continue
        
        filtered.append(c)
    
    logger.info(f"After filtering: {len(filtered)} candidates (from {len(candidates)})")
    return filtered


def to_seed_yaml_candidates(candidates: list[dict]) -> str:
    """
    Format candidates as YAML for human review.
    
    Output is NOT directly usable as seed_datasets.yaml.
    A human must review, add adapter names, language codes, etc.
    """
    output_lines = [
        "# AUTO-GENERATED DISCOVERY CANDIDATES",
        "# Review these and add to seed_datasets.yaml manually",
        "# DO NOT commit this file as-is",
        "#",
        "# Fields to fill in for each dataset:",
        "#   adapter: CommonVoiceAdapter | FLEURSAdapter | ShrutilipiAdapter | GenericHFAdapter",
        "#   languages: [hi, bn, ta, ...]  # BCP-47 codes",
        "#   license: cc0-1.0 | cc-by-4.0 | ...",
        "#   quality_tier: gold | silver | bronze | unknown",
        "",
        "candidates:",
    ]
    
    for c in sorted(candidates, key=lambda x: x.get("downloads", 0), reverse=True):
        output_lines.extend([
            f"  - id: {c['id']}",
            f"    # found_via: {c.get('found_via', 'unknown')}",
            f"    # downloads: {c.get('downloads', 0)}, likes: {c.get('likes', 0)}",
            f"    # tags: {c.get('tags', [])}",
            f"    hf_repo_id: {c['id']}",
            f"    name: \"TODO: Add human-readable name\"",
            f"    adapter: GenericHFAdapter  # TODO: update",
            f"    languages: []  # TODO: fill in BCP-47 codes",
            f"    license: unknown  # TODO: check",
            f"    quality_tier: unknown  # TODO: assess",
            f"    enabled: false  # TODO: set true when ready",
            "",
        ])
    
    return "\n".join(output_lines)


def main():
    parser = argparse.ArgumentParser(
        description="Discover Indic ASR datasets on HuggingFace Hub"
    )
    parser.add_argument(
        "--output",
        default="discovered_candidates.yaml",
        help="Output file for discovered candidates"
    )
    parser.add_argument(
        "--orgs",
        nargs="+",
        default=None,
        help="Organizations to search (default: all trusted orgs)"
    )
    parser.add_argument(
        "--langs",
        nargs="+",
        default=None,
        help="Language codes to search (default: all Indic)"
    )
    parser.add_argument(
        "--keywords",
        nargs="+",
        default=None,
        help="Keywords to search (default: all Indic ASR keywords)"
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Skip heuristic filtering of candidates"
    )
    args = parser.parse_args()

    candidates = discover_all(
        orgs=args.orgs,
        keywords=args.keywords,
        langs=args.langs,
    )

    if not args.no_filter:
        candidates = filter_candidates(candidates)

    yaml_output = to_seed_yaml_candidates(candidates)

    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(yaml_output)

    logger.info(f"Candidates written to: {output_path}")
    logger.info(f"Review and manually add to seed_datasets.yaml")


if __name__ == "__main__":
    main()
