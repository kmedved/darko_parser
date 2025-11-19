# Agent Instructions

This repository must always maintain compatibility with both the **playbyplayv2** API data and the **CDN** API data.
The project is now fully standalone and does not depend on, or target integration with, `kmedved/nba_scraper` or `kmedved/nba_parser`.

Agents must never guess about data structures or responses and must never use dummy or fabricated data.

## Repository file overview
This section documents every tracked file in the repository so downstream LLM agents can quickly locate code, tests, and fixtures.

### Top-level metadata and configuration
* `README.md` – User-facing package overview, installation/usage examples, schema notes, and environment variable knobs for parsing, mapping, and lineup enrichment.
* `CHANGELOG.md` – Release notes and unreleased changes, including parser compatibility updates and maintainer handoff details.
* `LICENSE` – GPLv3 license text for the project.
* `requirements.txt` – Runtime dependencies pinned for the package (numpy, pandas, requests, etc.).
* `setup.py` / `setup.cfg` / `MANIFEST.in` – Packaging configuration for PyPI distribution and included data files.

### Core package (`nba_scraper/`)
* `__init__.py` – Re-exports the primary scraping helpers (`scrape_game`, `scrape_season`, `scrape_date_range`, `scrape_from_files`).
* `nba_scraper.py` – Public entry points wrapping the unified parsing pipeline, CSV/pandas output selection, date validation, and filesystem output handling.
* `scrape_functions.py` – Minimal façade calling `io_sources.parse_any` against CDN remote feeds with optional YAML mapping overrides.
* `io_sources.py` – Source router that loads JSON from remote CDN, local fixtures, or in-memory dicts; coordinates optional shot chart backfill, lineup attachment, and boxscore validation based on environment flags; exposes `SourceKind` enum and helper loaders.
* `cdn_client.py` – HTTP client for CDN play-by-play/boxscore/shotchart/schedule endpoints with retry-enabled `requests.Session` helpers.
* `helper_functions.py` – Utility helpers for time parsing (`iso_clock_to_pctimestring`, `seconds_elapsed`), date-to-season derivation, and CDN schedule range querying.
* `cdn_parser.py` – Canonical parser for CDN liveData payloads: normalizes descriptors, computes event/family metadata, links sidecar actions (steals/blocks), enriches qualifiers, synthesizes coordinates when missing, and returns sorted canonical dataframes.
* `v2_parser.py` – Legacy JSON archive parser that mirrors canonical columns, infers event families/subfamilies, fills missing coordinates, and applies YAML overrides similar to the CDN parser.
* `parser_utils.py` – Shared helpers for both parsers (team field backfilling, shot-coordinate synthesis, possession inference, dataframe finalization, and coordinate presets).
* `lineup_builder.py` – Lineup reconstruction utilities: seeds starters from box scores, processes substitution events (CDN and v2 semantics), tracks on-court player IDs/names, and backfills lineup columns per event.
* `coords_backfill.py` – Merges shot chart data into parsed play-by-play frames, replacing synthesized/missing coordinates and cleaning style flags.
* `boxscore_validation.py` – Computes team totals from canonical PbP, compares against official box scores, and provides logging helpers plus field constant tuples.
* `helper_functions.py` – See above; located in core package to support scraping flows.
* `schema.py` – Canonical column ordering, event type labels, and simple numeric helpers (`int_or_zero`, `scoremargin_str`, `points_made_from_family`).
* `nba_scraper/stats/glossary_schema.csv` – Canonical schema for the per-player box returned by `PbP.player_box_glossary()` / `player_box_glossary_with_totals()`. Columns: `Column`, `Type`, `Example`, `Definition`. Downstream agents should treat this as the single source of truth for what each column means.

### Mapping utilities (`nba_scraper/mapping/`)
* `__init__.py` – Re-exports descriptor normalization and event code helpers.
* `descriptor_norm.py` – Normalizes descriptor strings, extracts style flags, and provides canonicalization helpers for mapping keys.
* `event_codebook.py` – Legacy event/family/subtype lookup tables plus functions to compute `eventmsgtype`, `eventmsgactiontype`, and free-throw trip counters (`ft_n_m`).
* `loader.py` – YAML loader that converts curated signature mappings into lookup dictionaries keyed by `(family, subType, descriptor_core, qualifiers)` tuples.
* `mapping_template.yml` – Example YAML showing how to override subfamily and msgaction/msgtype codes for specific signatures.

### Scripts
* `scripts/cataloguer.py` – Standalone CLI for cataloging CDN payload schemas: fetches and caches play-by-play/boxscore JSON, derives action signatures, summarizes field usage, and outputs mapping baselines to detect upstream data drift.

### Tests and fixtures (`tests/`)
* `test_boxscore_validation.py` – Verifies PbP-derived totals match box score fixtures, checks empty-input behavior, and exercises extended stat field comparisons.
* `test_functional.py` – Functional smoke tests across parsing pathways (CDN and v2) using fixture JSON, ensuring canonical columns and lineup enrichment are present.
* `test_integration.py` – Integration tests that round-trip parsed dataframes through `nba_parser` to validate compatibility and lineup/possession consistency.
* `test_mapping_overrides.py` – Ensures YAML mapping overrides remap event families/action codes as expected for CDN and v2 inputs.
* `test_unit.py` – Unit-level coverage for helper utilities (time parsing, possession inference, shot coordinate synthesis, etc.).
* Fixture files under `tests/test_files/` – Frozen CDN/v2 JSON payloads and minimal YAML mapping used by the test suite (`cdn_playbyplay_0022400001.json`, `cdn_boxscore_0022400001.json`, `cdn_shotchart_0022400001.json`, `v2_pbp_0021700001.json`, `mapping_min.yml`).
* Stats-focused regression tests live under `tests/stats/`. Use the shared fixtures defined in `tests/stats/conftest.py` (e.g., `v2_pbp_df`, `pbp_v2_game`, and `stats_fixture_dir`) instead of calling `pd.read_csv("test/...csv")` directly; all CSV fixtures reside in `tests/stats/test_files/`.

### Project data and misc
* `requirements.txt` – Dependency pins used for installation/testing.
* `setup.cfg` / `setup.py` / `MANIFEST.in` – Packaging metadata, pytest config, lint settings, and included files for distribution.
* `scripts/` – See entry above.

## How `nba_scraper.stats` Works
The parsing logic has been merged into this repository under `nba_scraper/stats/`.

### Usage
Instead of `nba_parser`, use the internal `PbP` class:

```python
from nba_scraper import scrape_game, lineup_builder
from nba_scraper.stats import PbP

df = scrape_game(["0021900151"])
df = lineup_builder.attach_lineups(df)

parser = PbP(df)
box_score = parser.player_box_glossary() # Returns player stats + team totals
rapm = parser.rapm_possessions()         # Returns possession-level data
```

`PbP.player_box_glossary()` returns columns defined in `glossary_schema.csv`. When
describing metrics (TSAttempts, USG, ASTpct, etc.), consult this CSV instead of
guessing. Do not invent new columns or semantics; if something is missing or unclear,
flag it for human review.
