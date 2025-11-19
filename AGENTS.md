# Agent Instructions

This repository must always maintain compatibility with both the **playbyplayv2** API data and the **CDN** API data.
The goal for any changes is to produce data that can be parsed and read by `kmedved/nba_parser`.

Agents must never guess about data structures or responses and must never use dummy or fabricated data.

## Repository file overview
This section documents every tracked file in the repository so downstream LLM agents can quickly locate code, tests, and fixtures.

### Top-level metadata and configuration
* `README.md` – User-facing package overview, installation/usage examples, schema notes, and environment variable knobs for parsing, mapping, and lineup enrichment.
* `CHANGELOG.md` – Release notes and unreleased changes, including parser compatibility updates and maintainer handoff details.
* `LICENSE` – GPLv3 license text for the project.
* `requirements.txt` – Runtime dependencies pinned for the package (numpy, pandas, requests, etc.).
* `setup.py` / `setup.cfg` / `MANIFEST.in` – Packaging configuration for PyPI distribution and included data files.
* `get_api_calls.py` – Obsolete stub raising a `RuntimeError` to redirect users to the public scraping API.

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
* `stat_calc_functions.py` – Small vectorized helpers for downstream stat calculations (shot made flag, foul/shot subtype parsing, points/seconds extraction).
* `helper_functions.py` – See above; located in core package to support scraping flows.
* `schema.py` – Canonical column ordering, event type labels, and simple numeric helpers (`int_or_zero`, `scoremargin_str`, `points_made_from_family`).

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
* `test_teamtotals_ft_metrics.py` – Checks free-throw trip metadata and team total calculations for correctness.
* `test_unit.py` – Unit-level coverage for helper utilities (time parsing, possession inference, shot coordinate synthesis, etc.).
* `test_with_parser.py` – Validates interactions with the external `nba_parser` package using canonical dataframe outputs.
* Fixture files under `tests/test_files/` – Frozen CDN/v2 JSON payloads and minimal YAML mapping used by the test suite (`cdn_playbyplay_0022400001.json`, `cdn_boxscore_0022400001.json`, `cdn_shotchart_0022400001.json`, `v2_pbp_0021700001.json`, `mapping_min.yml`).

### Project data and misc
* `requirements.txt` – Dependency pins used for installation/testing.
* `setup.cfg` / `setup.py` / `MANIFEST.in` – Packaging metadata, pytest config, lint settings, and included files for distribution.
* `scripts/` and `get_api_calls.py` – See entries above; `get_api_calls.py` intentionally errors to redirect users to supported APIs.

## How `nba_parser` Works (LLM-oriented overview)
This section explains how to feed data produced by the `nba_scraper` fork into `nba_parser`, which classes and helpers do the heavy lifting, and what the expected schemas look like. Treat it as a quick-start readme for future LLM agents.

### Expected play-by-play input
`nba_parser.PbP` consumes a single-game Pandas DataFrame that already contains lineup context (typically added via `nba_scraper.lineup_builder.attach_lineups`). The parser assumes the following columns are present and correctly typed:
- Game metadata: `game_id`, `season`, `game_date`, `home_team_abbrev`, `away_team_abbrev`, `home_team_id`, `away_team_id`. Dates may arrive as strings (CSV) or datetimes; the constructor coerces them into datetimes.
- Event ownership and scoring: `event_team` (matching a team abbrev), `event_type_de` (canonical family such as `shot`, `missed_shot`, `free-throw`, `turnover`, `foul`, `rebound`), `points_made`, `shot_made`, `is_three`, `is_block`, `is_steal`, `is_o_rebound`, `is_d_rebound`, `subfamily_de`/`subfamily` (for foul/turnover flavor), and `qualifiers` (used for detecting and-ones). Free-throw numbering text in `homedescription`/`visitordescription` is used to close possessions.
- Player participation: shooter/pass/steal/block IDs in `player1_id`, `player2_id`, `player3_id` with matching team IDs (`player1_team_id`, etc.), plus on-court lineups in `home_player_{1..5}_id` and `away_player_{1..5}_id`. Event timing `event_length` and `seconds_elapsed` are used for minutes and possession boundaries.
- Lineup-based possession context created by `nba_scraper`: `home_possession`/`away_possession` flags are derived in the constructor using shots, turnovers, defensive rebounds, and final free-throws.

### Core pipeline inside `PbP`
- **Initialization** (`PbP.__init__`): Normalizes date fields, casts `scoremargin` to string, and derives home/away possession markers that segment the play-by-play into possessions (`home_possession`/`away_possession`). The possession logic keys off made shots, turnovers, defensive rebounds, and terminal free throws to ensure both CDN and `playbyplayv2` feeds are handled consistently.【F:nba_parser/pbp.py†L17-L88】
- **Event enrichment** (`box_glossary.annotate_events`): Standardizes event families, infers `team_id`, creates booleans for FGA/FGM/FTs, three-pointers, turnover type (live/dead), foul flavors, and and-ones, and buckets shots into distance zones (`0_3`, `4_9`, `10_17`, `18_23`). The helper also assigns offensive/defensive team IDs per event.【F:nba_parser/box_glossary.py†L1-L119】【F:nba_parser/box_glossary.py†L120-L214】
- **Per-player counting** (`accumulate_player_counts`): Walks enriched events to accumulate makes/attempts by zone, assisted vs. unassisted splits, free throws, rebounds, turnovers (live/dead), fouls drawn/committed, steals, blocks (with possession-after context), goaltends, and and-ones. Output is one row per `(game_id, team_id, player_id)`.【F:nba_parser/box_glossary.py†L216-L330】
- **On-court exposure** (`compute_on_court_exposures`): Uses lineup columns and `event_length` to assign minutes, on-court rebounding opportunities, and team scoring/attempt aggregates to each player. It also builds possessions via `_build_possessions` to attribute offensive/defensive possessions and on/off scoring to players on the floor. Minutes are reconciled with the parser’s time-on-court calculation when possible.【F:nba_parser/box_glossary.py†L332-L438】【F:nba_parser/box_glossary.py†L440-L520】
- **Possession parsing** (`PbP._build_possessions` and `PbP.parse_possessions`): Segments the play-by-play between successive possession flags, labels offensive/defensive sides, and (optionally) aggregates shooting stats per possession for both teams. This powers RAPM exports and on/off calculations.【F:nba_parser/pbp.py†L1789-L1886】【F:nba_parser/pbp.py†L1887-L1965】
- **Box assembly** (`build_player_box`): Merges counts and exposures, drops non-participants, fills metadata, and computes numerous rate stats (TS%, USG, OREB_FGA/FT, assisted/unassisted splits by zone, AST-by-zone rates, on/off scoring, etc.). It aligns outputs with glossary-friendly naming and preserves rows with on-court scoring even if minutes are zero to avoid breaking team totals.【F:nba_parser/box_glossary.py†L522-L736】

### Public APIs and their expectations
- `PbP.playerbygamestats()`: Returns basic shooting, rebounding, turnover, foul, steal, block, assist, and time-on-court counts per player for the single game. Relies on the enriched play-by-play columns noted above.【F:nba_parser/pbp.py†L90-L761】
- `PbP.teambygamestats()`: Aggregates similar stats at the team level, including possessions, points for/against, fouls drawn, shots blocked, and winner/home flags.【F:nba_parser/pbp.py†L763-L1102】
- `PbP.player_box_glossary()`: Builds a glossary-aligned per-player box score by orchestrating event annotation, counting, on/off exposure, and the box assembler. Optional `player_meta`/`game_meta` DataFrames can be merged for IDs/biographical context.【F:nba_parser/pbp.py†L1967-L2012】
- `PbP.rapm_possessions()`: Exports one row per possession with offensive/defensive team IDs, participating players, and scoring/attempt aggregates—used for RAPM modeling.【F:nba_parser/pbp.py†L1864-L1965】
- `TeamTotals`/`PlayerTotals`: Consume lists of per-game stats (team or player) to produce multi-game aggregates, advanced rate stats, and RAPM regression helpers. These classes expect their inputs to mirror the outputs of `teambygamestats()` or `playerbygamestats()` and rely on consistent possession counts across games.【F:nba_parser/pbp.py†L1118-L1533】

### Common pitfalls when wiring `nba_scraper` → `nba_parser`
- Ensure lineup columns (`home_player_{1..5}_id`, `away_player_{1..5}_id`) and `event_length`/`seconds_elapsed` are populated; missing values lead to zero-minute on-court rows or dropped possessions.
- Provide canonical `event_type_de` strings. The parser accepts both CDN and `playbyplayv2` shapes, but ambiguous families (e.g., missing `turnover` vs. `foul`) reduce counting accuracy.
- Keep player IDs as integers and `team_id` values consistent across shooter/passer/steal/block columns; mismatches can drop rows during aggregation.
- When reading CSV exports, let the constructor coerce `game_date` to datetime; passing mixed formats without conversion can affect downstream merges.

### Quick validation workflow
1. Scrape a game via `nba_scraper` (CDN or `playbyplayv2`), attach lineups, and instantiate `PbP` with the resulting DataFrame.
2. Call `player_box_glossary()` or `playerbygamestats()`/`teambygamestats()` and compare against the fixture CSVs under `test/` for the same `game_id`.
3. For possession-level sanity checks, call `rapm_possessions()` and confirm offensive/defensive team totals match summed play-by-play scoring.

### Minimal usage example
This snippet shows the smallest end-to-end flow using the bundled CDN-era fixture (`0021900151_cdn.csv`). Swap the CSV for a fresh scrape from `nba_scraper` when validating new games.

```
import pandas as pd
from nba_parser import PbP

# Load a single game's play-by-play (lineups already attached in the fixture)
pbp_df = pd.read_csv("test/0021900151_cdn.csv")

# Parse player-level glossary box scores
parser = PbP(pbp_df)
player_box = parser.player_box_glossary()

# Inspect the first few rows or export to CSV
print(player_box.head())
player_box.to_csv("player_box_0021900151.csv", index=False)
```
