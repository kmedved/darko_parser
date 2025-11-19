# CLAUDE.md - AI Assistant Guide for darko_parser

## Project Overview

**darko_parser** (published as `nba_scraper` on PyPI) is a Python package that scrapes and parses NBA play-by-play data from both:
- Modern **CDN live data feeds** (2019+)
- Legacy **v2 JSON archives** (pre-2019)

The project normalizes both data sources into a **canonical schema** with structured columns for analytics pipelines.

**Maintainer**: Kostya Medvedovsky (@kmedved, creator of DARKO)
**Original Author**: Matthew Barlowe
**License**: GPL v3.0
**Python**: 3.10+
**Repository**: https://github.com/kmedved/darko_parser

## Critical Principles

### 🚨 NEVER Guess Data Structures
- **Never fabricate or assume API response structures**
- Always use real fixture data from `tests/test_files/` for testing
- When uncertain about data formats, read existing parser code or test fixtures
- This is explicitly stated in AGENTS.md: "Agents must never guess about data structures or responses and must never use dummy or fabricated data"

### 🔄 Dual API Compatibility
- Every change must maintain compatibility with **both CDN and v2 data sources**
- Test changes against both `cdn_playbyplay_*.json` and v2 fixtures
- The canonical schema defined in `nba_scraper/schema.py` is the source of truth

### 🏗️ Standalone Architecture
- This project is **standalone** and no longer depends on external `nba_parser` or `nba_scraper` packages
- Stats functionality is bundled under `nba_scraper/stats/`
- All parsing happens within this repository

## Repository Structure

```
darko_parser/
├── nba_scraper/               # Core package
│   ├── __init__.py           # Public API exports
│   ├── nba_scraper.py        # High-level entry points (scrape_game, scrape_season, etc.)
│   ├── scrape_functions.py   # Minimal façade for CDN remote feeds
│   ├── io_sources.py         # Source router (CDN/v2, remote/local/dict)
│   ├── cdn_client.py         # HTTP client for CDN endpoints
│   ├── cdn_parser.py         # CDN liveData parser → canonical dataframe
│   ├── v2_parser.py          # Legacy v2 parser → canonical dataframe
│   ├── parser_utils.py       # Shared parsing utilities
│   ├── helper_functions.py   # Time/date utilities, season detection
│   ├── schema.py             # Canonical column definitions & helpers
│   ├── lineup_builder.py     # Reconstruct on-court lineups from events
│   ├── coords_backfill.py    # Merge shot chart coordinates
│   ├── boxscore_validation.py# Validate parsed PbP against box scores
│   ├── mapping/              # Event mapping & normalization
│   │   ├── __init__.py
│   │   ├── descriptor_norm.py    # Normalize descriptor strings
│   │   ├── event_codebook.py     # Legacy event/family lookup tables
│   │   ├── loader.py             # YAML mapping loader
│   │   └── mapping_template.yml  # Example YAML overrides
│   └── stats/                # Analytics layer (bundled)
│       ├── __init__.py
│       ├── pbp.py            # PbP class for analytics
│       └── box_glossary.py   # Player/team box score generation
├── tests/                    # Comprehensive test suite
│   ├── test_unit.py          # Unit tests for helpers
│   ├── test_functional.py    # Functional tests (CDN & v2 parsing)
│   ├── test_integration.py   # Integration tests with nba_parser
│   ├── test_mapping_overrides.py  # YAML mapping tests
│   ├── test_boxscore_validation.py # Box score validation tests
│   ├── test_files/           # Frozen JSON/CSV fixtures
│   │   ├── cdn_playbyplay_*.json
│   │   ├── cdn_boxscore_*.json
│   │   ├── cdn_shotchart_*.json
│   │   ├── v2_pbp_*.json
│   │   └── *.csv (expected outputs)
│   └── stats/                # Stats-specific tests
│       ├── conftest.py       # Shared test fixtures (USE THESE!)
│       ├── test_files/       # Stats CSV fixtures
│       └── test_*.py         # Stats test modules
├── scripts/
│   └── cataloguer.py         # CLI for cataloging CDN schemas
├── setup.py                  # Package configuration
├── requirements.txt          # Dev dependencies (-e .[dev])
├── README.md                 # User documentation
├── AGENTS.md                 # Agent-specific instructions
├── CHANGELOG.md              # Release history
└── CLAUDE.md                 # This file

```

## Canonical Schema

The **single source of truth** for output columns is `nba_scraper/schema.py`:
- `CANONICAL_COLUMNS`: List of all expected columns in order
- `EVENT_TYPE_DE`: Mapping of event types to human-readable labels
- Helper functions: `int_or_zero()`, `scoremargin_str()`, `points_made_from_family()`

### Key Columns
Both parsers produce these fields:
- **Event identification**: `eventnum`, `action_number`, `pctimestring`, `period`
- **Team context**: `event_team`, `home_team_abbrev`, `away_team_abbrev`
- **Player IDs & names**: `player1_id`, `player1_name`, `player1_team_id` (and player2, player3)
- **Event classification**: `family`, `subfamily`, `eventmsgtype`, `eventmsgactiontype`, `event_type_de`
- **Shot metadata**: `is_three`, `shot_made`, `points_made`, `shot_distance`, `x`, `y`, `side`, `area`
- **Possessions**: `possession_raw`, `possession_after`, `is_turnover`, `is_steal`, `is_block`
- **Free throws**: `ft_n`, `ft_m` (trip counters, e.g., "2 of 3")
- **Lineups**: `home_player_1`...`home_player_5`, `away_player_1`...`away_player_5` (IDs and names)
- **Score state**: `score_home`, `score_away`, `scoremargin`
- **Metadata**: `game_id`, `game_date` (YYYY-MM-DD), `season`

## Public API (High-Level)

Import via `nba_scraper`:
```python
from nba_scraper import scrape_game, scrape_season, scrape_date_range, scrape_from_files, PbP
```

### Functions

#### `scrape_game(game_ids, data_format='dataframe', data_dir=None)`
Scrape specific games by ID.
- **game_ids**: List of game IDs (10-digit strings or ints, leading zeros optional)
- **data_format**: `'dataframe'` (returns pandas.DataFrame) or `'csv'` (writes files)
- **data_dir**: Directory for CSV output (defaults to home directory)
- Fetches CDN play-by-play, box score, and (optionally) shot chart

#### `scrape_season(season, data_format='dataframe', data_dir=None)`
Scrape all regular-season games for a season.
- **season**: Four-digit year (int or string)
- Returns dataframe or writes CSV files

#### `scrape_date_range(start_date, end_date, data_format='dataframe', data_dir=None)`
Scrape regular-season games between two dates (inclusive).
- **start_date**, **end_date**: Strings in `YYYY-MM-DD` format
- Returns dataframe or writes CSV files

#### `scrape_from_files(pbp_path, box_path=None, kind='cdn_local', data_format='dataframe', data_dir=None)`
Parse local JSON fixtures.
- **kind**: `'cdn_local'` (requires both pbp and box paths) or `'v2_local'` (only pbp_path)
- Routes to `io_sources.parse_any()` internally

### Analytics: `PbP` Class
```python
from nba_scraper import PbP

df = scrape_game(["0021900151"])
parser = PbP(df)
box_score = parser.player_box_glossary()  # Player stats + team totals
rapm_data = parser.rapm_possessions()     # RAPM-ready possession data
```

## Low-Level API (io_sources)

For advanced control:
```python
from nba_scraper.io_sources import parse_any, SourceKind

# CDN remote
df = parse_any("0022400001", SourceKind.CDN_REMOTE)

# CDN local fixtures
df = parse_any("playbyplay_0022400001.json", SourceKind.CDN_LOCAL, boxscore_path="boxscore_0022400001.json")

# Legacy v2 local
df = parse_any("0021700001.json", SourceKind.V2_LOCAL)

# In-memory dict
df = parse_any(json_dict, SourceKind.V2_DICT)
```

## Environment Variables

### `NBA_SCRAPER_MAP`
Path to a YAML file (derived from `mapping/mapping_template.yml`) to override `subfamily` and `eventmsgactiontype` for specific event signatures.
- Parsers load at runtime but succeed if unset
- See `nba_scraper/mapping/loader.py` for structure

### `NBA_SCRAPER_SYNTH_FT_DESC`
Set to `"1"` to synthesize "Free Throw N of M" text in `homedescription` / `visitordescription` for legacy consumers.
- Default: off
- Structured `ft_n` / `ft_m` columns are always present

### `NBA_SCRAPER_BACKFILL_COORDS`
Set to `"1"` to fetch shot charts and merge spatial coordinates (`x`, `y`) for shot attempts.
- Default: off
- Uses synthesized coordinates if disabled

## Development Workflows

### Setup
```bash
# Clone and install in editable mode with dev dependencies
git clone https://github.com/kmedved/darko_parser.git
cd darko_parser
pip install -r requirements.txt
```

This installs the package in editable mode (`-e .[dev]`) with pytest, black, and coverage tools.

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=nba_scraper --cov-report=html

# Run specific test file
pytest tests/test_functional.py

# Run specific test
pytest tests/test_unit.py::test_iso_clock_to_pctimestring
```

### Test Fixtures
- **Always use fixtures from `tests/stats/conftest.py`** for stats tests:
  - `v2_pbp_df`: Parsed v2 dataframe
  - `pbp_v2_game`: PbP instance
  - `stats_fixture_dir`: Path to `tests/stats/test_files/`
- **Never read CSVs directly** with `pd.read_csv("test/...csv")` in tests
- CDN/v2 fixtures live in `tests/test_files/`:
  - `cdn_playbyplay_0022400001.json`
  - `cdn_boxscore_0022400001.json`
  - `cdn_shotchart_0022400001.json`
  - `v2_pbp_0021700001.json` (if exists)

### Code Style
- **Type hints**: Use `from __future__ import annotations` and type all function signatures
- **Docstrings**: Include for all public functions and complex internal helpers
- **Naming**: Use snake_case for functions/variables, PascalCase for classes
- **Error handling**: Gracefully handle missing/malformed data (return empty strings, zeros, or log warnings)
- **Black formatting**: Format code with `black` (configured in `setup.py` extras)

### Adding New Features

1. **Update both parsers** (`cdn_parser.py` and `v2_parser.py`) to maintain parity
2. **Add columns to `schema.py`** if introducing new fields
3. **Write tests** in appropriate test file(s):
   - Unit tests → `test_unit.py`
   - Functional (end-to-end parsing) → `test_functional.py`
   - Mapping overrides → `test_mapping_overrides.py`
   - Stats → `tests/stats/test_*.py`
4. **Use real fixtures**: Add new test JSON files under `tests/test_files/` if needed
5. **Update CHANGELOG.md** under "Unreleased" section
6. **Update README.md** if changing public API

### Common Tasks

#### Adding a new canonical column
1. Add to `CANONICAL_COLUMNS` in `nba_scraper/schema.py`
2. Populate in `cdn_parser.py`'s `parse_cdn_playbyplay()` (update row dict)
3. Populate in `v2_parser.py`'s `parse_v2_playbyplay()` (update row dict)
4. Write tests verifying column exists in both CDN and v2 outputs

#### Fixing an event classification bug
1. Check if it's a mapping issue (can be fixed via YAML in `NBA_SCRAPER_MAP`)
2. If codebook change needed:
   - Update `mapping/event_codebook.py`
   - Add test case in `test_mapping_overrides.py` or `test_unit.py`
3. If descriptor normalization issue:
   - Update `mapping/descriptor_norm.py`
   - Test with real fixture exhibiting the issue

#### Handling new CDN API fields
1. Use `scripts/cataloguer.py` to fetch and analyze new CDN payloads:
   ```bash
   python scripts/cataloguer.py --game-id 0022400001 --output catalog.json
   ```
2. Update `cdn_parser.py` to consume new fields
3. Add fixture JSON to `tests/test_files/`
4. Write test ensuring new fields are correctly parsed

## Git Workflow

### Branching
- Development happens on feature branches: `claude/claude-md-<session-id>`
- Main branch is default for PRs
- **Never push to a different branch without explicit permission**

### Commits
- Clear, descriptive commit messages
- Follow conventional commits style: `fix: ...`, `feat: ...`, `refactor: ...`, `test: ...`, `docs: ...`
- Recent commits show style:
  - "Fix technical free throw masking"
  - "Refine technical FT last-shot handling"

### Pull Requests
- PRs should include:
  - Summary of changes
  - Test plan
  - Reference to issues if applicable
- Ensure all tests pass before creating PR
- Update CHANGELOG.md under "Unreleased"

## Testing Strategy

### Test Hierarchy
1. **Unit tests** (`test_unit.py`): Test individual helper functions
   - Time parsing (`iso_clock_to_pctimestring`, `seconds_elapsed`)
   - Schema helpers (`int_or_zero`, `scoremargin_str`)
   - Possession inference
   - Shot coordinate synthesis

2. **Functional tests** (`test_functional.py`): End-to-end parsing
   - CDN fixtures → canonical dataframe
   - v2 fixtures → canonical dataframe
   - Verify canonical columns present
   - Verify lineup enrichment

3. **Integration tests** (`test_integration.py`): Cross-package compatibility
   - Round-trip through `nba_parser.PbP`
   - Verify lineup/possession consistency

4. **Mapping tests** (`test_mapping_overrides.py`): YAML overrides
   - Custom event family/subfamily mappings
   - Action type code overrides

5. **Validation tests** (`test_boxscore_validation.py`): Data integrity
   - PbP-derived totals match official box scores
   - Extended stat field comparisons

6. **Stats tests** (`tests/stats/`): Analytics layer
   - Box score generation (`test_box_glossary.py`)
   - RAPM possessions (`test_rapm_possessions.py`)
   - Integration with PbP class (`test_pbp_integration.py`)

### Key Test Patterns

**Good**:
```python
# Use conftest fixtures
def test_player_box(pbp_v2_game):
    result = pbp_v2_game.player_box_glossary()
    assert "player_name" in result.columns
```

**Bad**:
```python
# Don't read CSVs directly
def test_player_box():
    df = pd.read_csv("test/21900151.csv")  # ❌ DON'T DO THIS
    parser = PbP(df)
    ...
```

## Common Pitfalls for AI Assistants

### ❌ Don't:
1. **Assume API structure** - Always check fixtures or existing code
2. **Break dual compatibility** - Test both CDN and v2 after changes
3. **Add columns without updating `schema.py`** - Canonical schema is enforced
4. **Skip tests** - Every change needs test coverage
5. **Hard-code paths** - Use environment variables or function parameters
6. **Ignore type hints** - All new code should be fully typed
7. **Read test CSVs directly** - Use `conftest.py` fixtures in stats tests
8. **Modify `setup.py` version** - Version bumps happen during release, not in feature PRs
9. **Push to wrong branch** - Always push to the branch specified in git development instructions
10. **Create documentation without request** - Only update docs when explicitly asked or required by code changes

### ✅ Do:
1. **Read existing parsers** to understand patterns before adding features
2. **Check AGENTS.md** for repository-specific guidelines
3. **Use real fixtures** from `tests/test_files/` for testing
4. **Update CHANGELOG.md** for all user-facing changes
5. **Run full test suite** (`pytest`) before committing
6. **Follow type hint conventions** (`from __future__ import annotations`)
7. **Validate against box scores** when changing aggregation logic
8. **Check environment variable usage** for runtime configuration
9. **Use `io_sources.parse_any()`** as the low-level entry point
10. **Preserve backward compatibility** - many downstream users depend on canonical schema

## Key Files Reference

### Must-read for understanding the codebase:
- `README.md` - User documentation, API examples, schema overview
- `AGENTS.md` - Agent-specific instructions (no guessing data!)
- `nba_scraper/schema.py` - Canonical column definitions
- `nba_scraper/io_sources.py` - Source routing logic
- `nba_scraper/cdn_parser.py` - CDN parsing implementation
- `nba_scraper/v2_parser.py` - Legacy v2 parsing implementation

### When working on specific features:
- **Lineups**: `lineup_builder.py`, `test_functional.py`
- **Shot coordinates**: `coords_backfill.py`, `parser_utils.py` (`_synth_xy`)
- **Event mapping**: `mapping/loader.py`, `mapping/event_codebook.py`, `mapping/descriptor_norm.py`
- **Validation**: `boxscore_validation.py`, `test_boxscore_validation.py`
- **Stats**: `stats/pbp.py`, `stats/box_glossary.py`, `tests/stats/`
- **HTTP client**: `cdn_client.py`
- **Time/date utils**: `helper_functions.py`

## Debugging Tips

### Issue: Parsed dataframe missing expected columns
1. Check if column is in `schema.py::CANONICAL_COLUMNS`
2. Verify both `cdn_parser.py` and `v2_parser.py` populate the column
3. Run functional tests to see which parser is failing

### Issue: Event classification wrong
1. Check `mapping/event_codebook.py` for family/type mappings
2. Try YAML override via `NBA_SCRAPER_MAP` environment variable
3. Examine descriptor normalization in `mapping/descriptor_norm.py`
4. Add test case with the problematic event in `test_mapping_overrides.py`

### Issue: Box score totals don't match
1. Run `boxscore_validation.py` logic to see which fields differ
2. Check if events are missing `team_id` or `event_team` (required for aggregation)
3. Verify possession assignment logic in `parser_utils.py`
4. Review shot classification (`is_three`, `shot_made`, `points_made`)

### Issue: Lineup reconstruction incorrect
1. Check `lineup_builder.py` substitution handling
2. Verify starter detection from box score
3. Ensure CDN vs v2 substitution semantics are handled differently
4. Run integration tests to catch lineup/possession mismatches

### Issue: Test fixture outdated
1. Fetch fresh JSON using `cdn_client.py` helpers or `scripts/cataloguer.py`
2. Save to `tests/test_files/` with appropriate naming
3. Regenerate expected CSV by running parser and inspecting output
4. Update test assertions if schema changed

## Release Process

1. Update version in `setup.py`
2. Move "Unreleased" section in `CHANGELOG.md` to versioned release
3. Commit: `git commit -m "Release vX.Y.Z"`
4. Tag: `git tag vX.Y.Z`
5. Push: `git push && git push --tags`
6. Build: `python setup.py sdist bdist_wheel`
7. Upload to PyPI: `twine upload dist/*`

(Note: As an AI assistant, you typically won't perform releases, but this context is useful for understanding versioning)

## Additional Resources

- **PyPI Package**: https://pypi.org/project/nba-scraper/
- **GitHub Issues**: https://github.com/kmedved/darko_parser/issues
- **Maintainer Twitter**: @kmedved
- **Original Author**: Matthew Barlowe (thank you!)

## Summary for AI Assistants

When working on this repository:
1. **Never guess** - Read fixtures, existing code, or ask for clarification
2. **Test both APIs** - CDN and v2 must work identically
3. **Follow the schema** - `schema.py` is the source of truth
4. **Use real data** - No dummy or fabricated test data
5. **Read AGENTS.md** - Repository-specific constraints are documented there
6. **Maintain backwards compatibility** - Canonical schema is a stable interface
7. **Type everything** - Modern Python with full type hints
8. **Test thoroughly** - Unit, functional, integration, validation

This is a well-maintained analytics package with downstream users. Changes should be careful, well-tested, and preserve the canonical schema contract.
