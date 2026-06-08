# Indic ASR Ecosystem

A Python toolkit for aggregating, normalizing, and generating training-ready manifests from Indic language Automatic Speech Recognition (ASR) datasets hosted on HuggingFace Hub.

## What This Project Does

Training ASR models for Indic languages requires juggling dozens of datasets, each with different schemas, column names, split conventions, license terms, and quality levels. This project solves that by providing:

- A **curated catalogue** of known Indic ASR datasets (`catalogue/seed_datasets.yaml`)
- **Dataset adapters** that normalize each source into a canonical schema
- A **registry** of validated dataset entries in a stable JSON format
- A **manifest generator** that produces deduplicated, quality-filtered Parquet files ready for model training
- CLI scripts to **discover** new datasets on HuggingFace Hub and **ingest** them into the registry

Supported languages: Hindi, Bengali, Tamil, Telugu, Kannada, Malayalam, Marathi, Gujarati, Punjabi, Urdu, Odia, Assamese, Nepali, Sanskrit, Maithili, Bhojpuri, Konkani, Sindhi, Sinhala, Dzongkha.

---

## Architecture: Four-Layer Pipeline

```
Layer 0: seed_datasets.yaml        (human-maintained catalogue)
            ↓ ingest pipeline
Layer 1: Adapters                  (per-source schema translators)
            ↓
Layer 2: registry/entries/*.json   (validated, stable registry entries)
            ↓ manifest generator
Layer 3: manifests/*.parquet       (model-ready, deduplicated manifests)
```

**You only manually edit Layer 0.** Everything else is generated.

---

## Installation

```bash
# Clone the repo
git clone <repo-url>
cd indic_asr_ecosystem

# Install core dependencies
pip install -e .

# Install dev tools (pytest, ruff, mypy)
pip install -e ".[dev]"

# Install optional audio processing tools (for fingerprint deduplication)
pip install -e ".[audio]"
```

**Requirements:** Python 3.10+, PyTorch/CUDA not required (the toolkit is data-prep only).

---

## Quick Start

### 1. Dry-run the ingest pipeline

```bash
# Simulate ingesting all enabled datasets (no writes)
python scripts/ingest.py --dry-run

# Ingest only one dataset
python scripts/ingest.py --ids mozilla-foundation/common_voice_17_0 --dry-run

# Ingest only Hindi data
python scripts/ingest.py --langs hi --dry-run
```

### 2. Run the real ingest (writes to `registry/entries/`)

```bash
python scripts/ingest.py
```

### 3. Discover new datasets on HuggingFace Hub

```bash
python scripts/discover.py --output discovered_candidates.yaml
```

Review the output YAML and manually add entries you want to `catalogue/seed_datasets.yaml`.

### 4. Run tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
indic_asr_ecosystem/
├── catalogue/
│   └── seed_datasets.yaml          # The ONLY file you manually edit
├── indic_asr/
│   ├── schema.py                   # Pydantic models for all data layers
│   ├── adapters/
│   │   ├── base.py                 # Abstract base adapter
│   │   ├── common_voice.py         # Mozilla Common Voice adapter
│   │   ├── fleurs.py               # Google FLEURS adapter
│   │   └── generic_hf.py           # Generic + Shrutilipi + IndicTTS adapters
│   ├── registry/
│   │   ├── builder.py              # Runs adapters → writes JSON entries
│   │   ├── validator.py            # Semantic validation of registry entries
│   │   └── deduplicator.py         # Cross-dataset deduplication logic
│   └── manifest/
│       └── generator.py            # Registry → Parquet manifests
├── scripts/
│   ├── discover.py                 # HuggingFace Hub dataset discovery
│   └── ingest.py                   # Main ingest CLI
├── registry/                       # Auto-generated (do not edit manually)
│   └── entries/                    # One JSON file per (dataset, language, split)
├── manifests/                      # Auto-generated Parquet + sidecar JSON
├── tests/
│   └── test_schema_and_adapters.py
└── pyproject.toml
```

---

## Detailed Component Guide

### `catalogue/seed_datasets.yaml` — The Human-Maintained Catalogue

This is the single source of truth. Every dataset you want to ingest must have an entry here.

**Schema per entry:**

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique stable identifier (usually `org/repo`) |
| `name` | string | Human-readable name |
| `adapter` | string | Adapter class to use for this dataset |
| `source_type` | enum | `huggingface` \| `openslr` \| `custom_url` \| `local` |
| `hf_repo_id` | string | HuggingFace repo ID |
| `hf_configs` | list\|null | HF config names (language configs). `null` = auto-detect |
| `languages` | list | BCP-47 language codes covered |
| `license` | string | SPDX license identifier |
| `quality_tier` | enum | `gold` \| `silver` \| `bronze` \| `unknown` |
| `enabled` | bool | `false` entries are skipped during ingest |
| `notes` | string | Free-form human notes |

**Adding a new dataset — workflow:**

```yaml
# Step 1: Add entry with enabled: false
- id: "org/new-dataset"
  name: "New Dataset"
  adapter: GenericHFAdapter
  source_type: huggingface
  hf_repo_id: "org/new-dataset"
  hf_configs: null
  languages: [hi]
  license: cc-by-4.0
  quality_tier: silver
  enabled: false
  notes: "Verify before enabling"
```

```bash
# Step 2: Dry-run to check it parses correctly
python scripts/ingest.py --dry-run --ids org/new-dataset

# Step 3: If good, set enabled: true and ingest
python scripts/ingest.py --ids org/new-dataset
```

**Currently included datasets:**

| Dataset | Quality | Languages | Status |
|---------|---------|-----------|--------|
| Mozilla Common Voice 17.0 | Silver | hi, bn, mr, ta, te, kn, ml, ur, gu, pa, or, as, ne, sa | Enabled |
| Google FLEURS | Gold | hi, bn, as, gu, mr, pa, ur, or, ta, te, kn, ml, ne | Enabled |
| AI4Bharat Shrutilipi | Silver | hi, mr, ta, te, kn, ml, gu, bn, or, pa, as, ur | Enabled |
| AI4Bharat IndicSUPERB | Gold | hi, bn, as, gu, mr, pa, or, ta, te, kn, ml, ne, ur, sa | Enabled |
| Kathbath (IIT Madras) | Gold | hi, bn, as, gu, mr, pa, or, ta, te, kn, ml | Disabled (needs verification) |
| SMC Malayalam ASR Corpus | Silver | ml | Disabled |
| Granth Sanskrit (IIT Madras) | Gold | sa | Disabled (research-only license) |
| OpenSLR Indic Resources | Silver | hi, ta, te, kn, ml, gu, mr, bn | Disabled (needs custom adapter) |

---

### `indic_asr/schema.py` — Data Models

All schemas are **Pydantic v2 models** with strict validation. Language codes are auto-normalized to BCP-47.

#### `SeedDatasetEntry` (Layer 0)
Mirrors one entry in `seed_datasets.yaml`. Validated on load.

#### `RegistryEntry` (Layer 2)
One entry per `(dataset, language, split)` triple. Contains:
- Classification: language, split, domain
- HF access fields: repo ID, config, split name, audio/transcript column names
- Quality and licensing metadata
- Audio statistics (`AudioStats`)
- Full provenance chain (`ProvenanceRecord`)

Stored as `registry/entries/{entry_id}.json`. The `entry_id` format is `{source_id}__{lang}__{split}` (e.g., `mozilla-foundation__common_voice_17_0__hi__train`).

#### `ManifestRow` (Layer 3)
One row per utterance in the final Parquet manifest. Contains:
- A stable `utterance_id`
- Streaming pointers to HF Hub (repo, config, split, row index) — **no audio bytes**
- Raw and normalized transcripts
- Language, quality tier, license
- Audio fingerprint for deduplication
- Back-pointer to the source registry entry

#### `ManifestMetadata` (Layer 3 sidecar)
Stored alongside each Parquet as `{lang}_{split}_metadata.json`. Records how many rows, total duration, which registry entries were used, and what filters were applied.

#### Helper utilities
- `INDIC_LANGUAGE_CODES` — dict of all 20 supported BCP-47 codes → language names
- `LANGUAGE_ALIASES` — normalizes `"hindi"`, `"hi-IN"`, `"hin"` → `"hi"`, etc.
- `normalize_language_code(raw)` — normalizes any variant to canonical BCP-47

---

### `indic_asr/adapters/` — Dataset Adapters

Each adapter translates one source dataset's schema into canonical `RegistryEntry` objects. Adapters are:
- **Stateless** (no side effects)
- **Versioned** (bumping `VERSION` invalidates cached registry entries)
- **Network-aware** (use HF dataset builders without downloading audio)
- **Testable without network** (via mock fixtures)

#### `BaseAdapter` (abstract)

All adapters extend this. Provides:
- `iter_registry_entries(languages, splits, dry_run)` — the main entry point (abstract)
- `can_handle(seed_entry)` — class method; returns True if this adapter handles the given seed entry (abstract)
- `detect_audio_column(columns)` — finds the audio column by trying known variants
- `detect_transcript_column(columns)` — finds the transcript column
- `normalize_split(raw_split)` — normalizes `"dev"` → `VALIDATION`, `"eval"` → `TEST`, etc.
- `make_entry_id(language, split)` — generates stable `entry_id`
- `make_provenance(...)` — builds `ProvenanceRecord` with HF commit SHA, schema hash, etc.
- `compute_schema_hash(features)` — SHA256 of dataset schema for drift detection

#### `CommonVoiceAdapter`

Handles Mozilla Common Voice (all versions, v6–v17+). Key behavior:
- One HF config per language (e.g., `config="hi"` for Hindi)
- Maps CV-specific config names (`"pa-IN"`, `"ne-NP"`) to BCP-47
- Audio format: MP3 @ 32kbps
- Quality tier: Silver (crowd-sourced with community validation)
- License: CC0 (public domain)

#### `FLEURSAdapter`

Handles Google FLEURS. Key behavior:
- Config names use underscore format: `"hi_in"`, `"ta_in"`, `"ur_pk"`, etc.
- Audio: 16kHz WAV
- Quality tier: Gold (carefully curated, 10 speakers per language, ~12h each)
- License: CC-BY 4.0
- **Important:** FLEURS test split is the standard ASR benchmark — do not include it in training data.

#### `GenericHFAdapter`

Fallback adapter for well-structured HF datasets. Use this when the dataset has:
- Standard column names (`audio`, `sentence`, `transcription`, etc.)
- Standard split names (with automatic fallback: `dev`→`validation`, `eval`→`test`)
- One config per language (or no configs at all)

Also handles split name fallbacks automatically: if `validation` isn't found, tries `dev`, `valid`, `val`.

#### `ShrutilipiAdapter` (extends `GenericHFAdapter`)

AI4Bharat Shrutilipi broadcast news corpus. Overrides `domain="broadcast_news"` and forces `quality_tier=SILVER` for all entries.

#### `IndicTTSAdapter` (extends `GenericHFAdapter`)

AI4Bharat IndicTTS studio-recorded corpus. Overrides `domain="studio_read_speech"` and forces `quality_tier=GOLD`.

**Choosing the right adapter:**

| Source dataset type | Use adapter |
|---------------------|-------------|
| Mozilla Common Voice (any version) | `CommonVoiceAdapter` |
| Google FLEURS | `FLEURSAdapter` |
| AI4Bharat Shrutilipi | `ShrutilipiAdapter` |
| AI4Bharat IndicTTS | `IndicTTSAdapter` |
| Any other HF dataset with standard structure | `GenericHFAdapter` |
| Novel dataset needing custom logic | Write a new adapter extending `BaseAdapter` |

---

### `indic_asr/registry/` — Registry Layer

#### `RegistryBuilder`

Orchestrates the ingest pipeline: reads `seed_datasets.yaml`, selects the right adapter for each seed entry, runs `iter_registry_entries()`, validates the output, and writes JSON files atomically.

Key behaviors:
- **Idempotent:** Re-running skips unchanged entries (detected via adapter version + schema hash + HF commit SHA)
- **Atomic writes:** Uses temp file + `os.replace()` to avoid partial writes
- **Dry-run mode:** Logs what would be written without touching disk
- **Filtering:** Accepts `filter_languages`, `filter_splits`, `filter_ids` to ingest subsets

#### `RegistryValidator`

Two-level validation before any entry is written:

1. **Structural** — Pydantic handles this automatically on construction
2. **Semantic** — Business-rule checks:
   - `entry_id` matches the `language` and `split` fields
   - Language is a known Indic BCP-47 code
   - License is not `UNKNOWN` (configurable warning)
   - `AudioStats` fields are internally consistent (mean ≈ total/count)
   - Sample rate is a recognized ASR rate (8kHz, 16kHz, 22kHz, 44kHz, 48kHz)
   - HF fields are populated for HuggingFace-sourced entries
   - Provenance fields are complete

Errors block the write; warnings are logged and allow the write.

#### `Deduplicator`

Cross-dataset deduplication applied at manifest generation time (not at registry write time — the registry is always the full truth).

**`TranscriptDeduplicator`** (fast first pass):
1. Normalize transcript: NFC → lowercase → remove ASCII punctuation → collapse whitespace
2. SHA256 the result
3. When duplicate found: keep the higher `quality_tier` entry (GOLD > SILVER > BRONZE > UNKNOWN)

**`AudioFingerprintDeduplicator`** (accurate, expensive):
- SHA256 of the first 2 seconds of raw PCM audio (resampled to 16kHz, quantized to 16-bit)
- Robust to format differences (MP3 vs WAV vs FLAC of the same recording)
- Not robust to re-encodings or different recordings of the same text

**`deduplicate_manifest_rows(rows, strategy)`**: Top-level function. Groups rows by language first (no cross-language deduplication), then applies `TranscriptDeduplicator` per language group.

---

### `indic_asr/manifest/generator.py` — Manifest Generator

Reads all `registry/entries/*.json` files and builds model-ready Parquet files.

#### `ManifestFilters`

Configurable quality gate applied per row before writing to Parquet:

| Filter | Default | Description |
|--------|---------|-------------|
| `min_duration_s` | 0.5s | Drop utterances shorter than this |
| `max_duration_s` | 30.0s | Drop utterances longer than this |
| `min_transcript_chars` | 2 | Drop very short transcripts |
| `max_transcript_chars` | 500 | Drop very long transcripts |
| `allowed_licenses` | None (all) | Restrict to specific licenses |
| `allowed_quality_tiers` | None (all) | Restrict to GOLD/SILVER/etc. |
| `exclude_synthetic` | False | Drop synthetic/TTS data |
| `exclude_noisy` | False | Drop entries marked as noisy |

#### `normalize_transcript(text, language)`

Conservative normalization for acoustic model training:
- Unicode NFC (preserves Devanagari, Bengali, Tamil, etc. character compositions)
- Strip leading/trailing whitespace, collapse internal whitespace
- Remove ASCII punctuation that doesn't affect pronunciation
- Does **not** remove numerals, diacritics, or Indic-specific punctuation

Indic script Unicode ranges handled: Devanagari, Bengali/Assamese, Gurmukhi, Gujarati, Odia, Tamil, Telugu, Kannada, Malayalam, Arabic/Urdu.

#### `ManifestGenerator`

```python
from indic_asr.manifest.generator import ManifestGenerator, ManifestFilters
from indic_asr.schema import SplitType

generator = ManifestGenerator(
    registry_dir="registry/",
    output_dir="manifests/",
    filters=ManifestFilters(
        allowed_quality_tiers=["gold", "silver"],
        exclude_synthetic=True,
    ),
    deduplicate=True,
)

# Build manifest for Hindi training data
generator.build_manifest(language="hi", split=SplitType.TRAIN)

# Build all manifests for all languages and splits
generator.build_all_manifests()
```

Output: `manifests/hi_train.parquet` + `manifests/hi_train_metadata.json`

**Two row generation paths:**
- **Fast path** (`_generate_pointer_rows`): When row count is known from registry stats, generates pointer-only rows without streaming. Transcripts are fetched lazily by the training pipeline.
- **Full path** (`_generate_rows_with_inspection`): Streams through the source dataset to fetch transcripts. Required for transcript-level deduplication.

**Consuming manifests in training:**

```python
from datasets import load_dataset

ds = load_dataset("parquet", data_files={"train": "manifests/hi_train.parquet"})
# Stream audio lazily via the pointer fields:
#   audio_hf_repo, audio_hf_config, audio_hf_split, audio_hf_row_index
```

---

### `scripts/discover.py` — Dataset Discovery

Searches HuggingFace Hub for Indic ASR datasets using three strategies:
1. **Organization search** — queries known orgs: `ai4bharat`, `mozilla-foundation`, `google`, `openslr`, `iitm-lab`, `IISc-MILE`, `SMC`, etc.
2. **Keyword search** — 17 targeted keywords like `"indic asr"`, `"hindi speech"`, `"tamil speech"`, etc.
3. **Language tag search** — HF Hub `language` tags for all 15 Indic language codes

Results are deduplicated by dataset ID, heuristically filtered (minimum downloads/likes, must have audio-related tags), and written as a YAML candidate list for human review.

**The output is NOT directly usable as `seed_datasets.yaml`.** A human must review candidates, fill in adapter names, language codes, license, and quality tier.

```bash
# Full discovery
python scripts/discover.py --output candidates.yaml

# Search only AI4Bharat and IIT Madras
python scripts/discover.py --orgs ai4bharat iitm-lab

# Search only Tamil and Telugu
python scripts/discover.py --langs ta te

# Skip heuristic filtering (include all results)
python scripts/discover.py --no-filter
```

---

### `scripts/ingest.py` — Ingest Pipeline CLI

The main entry point for populating the registry.

```bash
# Ingest all enabled datasets in seed_datasets.yaml
python scripts/ingest.py

# Ingest specific dataset(s)
python scripts/ingest.py --ids mozilla-foundation/common_voice_17_0

# Filter to specific languages
python scripts/ingest.py --langs hi bn ta

# Filter to specific splits
python scripts/ingest.py --splits train validation

# Dry run (no disk writes)
python scripts/ingest.py --dry-run

# Force overwrite existing registry entries
python scripts/ingest.py --overwrite

# Verbose logging
python scripts/ingest.py --verbose

# Custom paths (for non-standard directory layouts)
python scripts/ingest.py --seed /path/to/catalogue.yaml --registry /path/to/registry/
```

**In Google Colab:**
```python
!python /content/indic_asr_ecosystem/scripts/ingest.py --dry-run
```

Exit code is non-zero if any dataset fails to ingest.

---

## Adding a New Adapter

If a new dataset has a non-standard schema, write a custom adapter:

```python
# indic_asr/adapters/my_dataset.py
from indic_asr.adapters.base import BaseAdapter
from indic_asr.schema import RegistryEntry, SeedDatasetEntry, SplitType
from typing import Iterator, Optional

class MyDatasetAdapter(BaseAdapter):
    VERSION = "1.0.0"  # Bump this when your logic changes

    @classmethod
    def can_handle(cls, seed_entry: SeedDatasetEntry) -> bool:
        return seed_entry.adapter == "MyDatasetAdapter"

    def iter_registry_entries(
        self,
        languages: Optional[list[str]] = None,
        splits: Optional[list[SplitType]] = None,
        dry_run: bool = False,
    ) -> Iterator[RegistryEntry]:
        repo_id = self.seed_entry.hf_repo_id
        commit_sha = self._get_hf_commit_sha(repo_id) if not dry_run else "dry-run"

        for lang in (languages or self.seed_entry.languages):
            builder = self._safe_load_builder(repo_id, lang)
            if builder is None:
                continue
            # ... your custom schema mapping logic ...
            yield RegistryEntry(
                entry_id=self.make_entry_id(lang, SplitType.TRAIN),
                # ... fill all required fields ...
                provenance=self.make_provenance(hf_commit_sha=commit_sha),
            )
```

Then register it in `indic_asr/adapters/__init__.py` so `get_adapter()` can find it.

---

## License Notes

Different datasets in the catalogue have different licenses. Key restrictions:

| License | Commercial use | Attribution required | Share-alike |
|---------|---------------|---------------------|-------------|
| CC0 (Common Voice) | Yes | No | No |
| CC-BY 4.0 (FLEURS, Shrutilipi, IndicSUPERB) | Yes | Yes | No |
| CC-BY-SA 4.0 | Yes | Yes | Yes (derivatives must be CC-BY-SA) |
| CC-BY-NC 4.0 | No | Yes | No |
| research-only (Granth Sanskrit) | No | Verify terms | — |

The `ManifestFilters.allowed_licenses` parameter lets you restrict manifests to datasets with compatible licenses for your use case.

---

## Running Tests

```bash
# All tests (no network required — adapters are mocked)
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=indic_asr --cov-report=term-missing
```

Tests cover:
- Language normalization (BCP-47 aliases, region stripping)
- Pydantic schema validation (seed entries, registry entries)
- Semantic registry validation (entry_id pattern, field consistency)
- Deduplication logic (exact duplicates, normalized duplicates, quality-based replacement, cross-language isolation)
- CommonVoiceAdapter with mocked HF builders (no network)

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `pydantic>=2.0` | Schema validation and serialization |
| `datasets>=2.14` | HuggingFace datasets library (streaming access) |
| `huggingface_hub>=0.20` | HF Hub API (discovery, commit SHA lookup) |
| `pyyaml>=6.0` | YAML parsing for seed catalogue |
| `pyarrow>=14.0` | Parquet read/write for manifests |
| `pandas>=2.0` | DataFrame operations |
| `librosa`, `soundfile`, `numpy` | *(optional)* Audio fingerprinting |
