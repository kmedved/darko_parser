[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Maintenance](https://img.shields.io/maintenance/yes/2024)](https://github.com/mcbarlowe/nba_scraper/commits/master)
[![PyPI version](https://badge.fury.io/py/nba-scraper.svg)](https://badge.fury.io/py/nba-scraper)
[![Downloads](https://pepy.tech/badge/nba-scraper)](https://pepy.tech/project/nba-scraper)
[![Build Status](https://travis-ci.org/mcbarlowe/nba_scraper.svg?branch=master)](https://travis-ci.org/mcbarlowe/nba_scraper)
[![codecov](https://codecov.io/gh/mcbarlowe/nba_scraper/branch/master/graph/badge.svg)](https://codecov.io/gh/mcbarlowe/nba_scraper)

# This project is actively maintained and now supports both legacy v2 and CDN (live data) endpoints out of the box.

> Maintained by Kostya Medvedovsky (@kmedved on Twitter, creator of DARKO). Immense thanks to Matthew Barlowe for building and sharing the original release of `nba_scraper`.
# `nba_scraper`

This is a package written in Python to scrape the NBA's official API (legacy v2
JSON archives and the modern CDN live data feeds) and produce the play by play
of games either in a `csv` file or a `pandas` dataframe. This package has two
main functions `scrape_game` which scrapes an individual game or a list of
specific games, and `scrape_season` which scrapes an entire season of regular
season games.

The scraper now supports both the modern NBA CDN live data feeds (2019 and later)
and legacy v2 JSON archives. No matter the source, results are normalised to a
canonical schema that includes structured shot metadata, team context and on-court
lineups computed from the play-by-play itself.

## Canonical dataframe columns

Each parsed play-by-play row exposes a common set of columns that are designed to
interoperate with downstream tools such as `nba_parser`. In addition to the core
fields (`eventmsgtype`, `pctimestring`, `player1_id`, etc.) the dataframe now
includes:

* `event_team` and `event_type_de` – a consistent team abbreviation and event
  label for every action.
* `player1_team_id`, `player2_team_id`, `player3_team_id` – acting and secondary
  team identifiers populated for assists, steals and blocks.
* `is_turnover`, `is_steal`, `is_block` – boolean flags used by possession
  aggregators.
* `eventnum` – stable event ordinal (alias of CDN `action_number` / V2 `eventnum`).
* `scoremargin` – home score minus away score as a string (compatible with legacy consumers).
* `ft_n` and `ft_m` – structured free-throw trip counters extracted from the
  feed and normalised for technical/single attempts.
* `game_date` – normalised to a `YYYY-MM-DD` string for consistent downstream handling.
* `season` – derived from the UTC tip-off date.
* `home_player_1`…`home_player_5` / `away_player_1`…`away_player_5` – lineup
  names alongside the existing ID columns.

These compatibility columns are present for both CDN and legacy v2 sources so
that the resulting dataframe can be dropped directly into analytics pipelines.

## YAML mapping

Set the ``NBA_SCRAPER_MAP`` environment variable to point at a curated YAML
file (derived from ``mapping_template.yml`` in the catalog) to refine
``eventmsgactiontype`` and ``subfamily`` classifications for turnovers, fouls
and violations. Parsers will load the mapping at runtime, but still succeed if
the variable is unset.

## Optional free-throw description synthesis

Legacy consumers that still rely on ``homedescription`` / ``visitordescription``
text can opt-in to synthesised “Free Throw N of M” strings by setting
``NBA_SCRAPER_SYNTH_FT_DESC=1``. The structured ``ft_n``/``ft_m`` counters
remain available regardless of this flag. Set the environment variable before
importing ``nba_scraper.cdn_parser`` if you need the override to apply during
module import.

## Using with `nba_parser`

The canonical dataframe can be passed straight into
[`nba_parser`](https://github.com/kmedved/nba_parser) for advanced possession and
box score calculations. `nba_scraper` now installs `nba_parser` directly from
the upstream default (`master`) branch to ensure you always have the latest
compatibility fixes:

```python
from pathlib import Path

from nba_scraper import io_sources, lineup_builder
from nba_parser import PbP

pbp_path = Path("cdn_playbyplay_0022400001.json")
box_path = Path("cdn_boxscore_0022400001.json")

df = io_sources.parse_any((pbp_path, box_path), io_sources.SourceKind.CDN_LOCAL)
df = lineup_builder.attach_lineups(df)

pbp = PbP(df)
player_totals = pbp.playerbygamestats()
team_totals = pbp.teambygamestats()
```

The example above uses local CDN fixtures, but any canonical dataframe returned
by `nba_scraper` (including legacy v2 games) will work with `nba_parser.PbP`.

# Installation

To install this package just type this at the command line:

    pip install nba_scraper

Advanced RAPM tooling depends on scikit-learn and is distributed as an optional
extra:

    pip install "nba_scraper[rapm]"

# Usage

`nba_scraper` exposes two levels of APIs:

* `nba_scraper.nba_scraper` – friendly, batteries-included helpers for the
  most common flows (`scrape_game`, `scrape_season`, `scrape_date_range`, etc.).
* `nba_scraper.io_sources.parse_any` – a low-level router you can use to point at
  CDN endpoints, cached JSON fixtures or already-in-memory dictionaries.

The sections below walk through both layers in a step-by-step fashion.

## Step 1 – Install and import

```python
import nba_scraper.nba_scraper as ns
from nba_scraper import io_sources
```

## Step 2 – Choose a data source

The scraper automatically normalises CDN and legacy v2 feeds to the same schema.
Pick the source that best matches your workflow:

| Scenario | Call | Notes |
| --- | --- | --- |
| Live or recently completed games via the official CDN API | `ns.scrape_game([...])` (default) or `io_sources.parse_any("0022400001", io_sources.SourceKind.CDN_REMOTE)` | Requires internet access. Automatically requests play-by-play, box score and (optionally) shot charts. |
| Downloaded CDN JSON fixtures | `ns.scrape_from_files(pbp_path, box_path, kind="cdn_local")` | Expects a pair of files – one `*playbyplay*.json` and one `*boxscore*.json`. |
| Archived legacy v2 JSON files | `ns.scrape_from_files(v2_path, kind="v2_local")` or `io_sources.parse_any(v2_path, io_sources.SourceKind.V2_LOCAL)` | Useful for historical games prior to the CDN feeds. |
| Already-loaded dictionaries | `io_sources.parse_any(payload_dict, io_sources.SourceKind.V2_DICT)` | Skips disk I/O if you have the JSON loaded elsewhere. |

When hitting the CDN endpoints directly you can pass full 10-digit game IDs
(`0022400001`) or the shortened form without leading zeros (`22400001`). The
helper automatically pads and validates the identifiers.

## Step 3 – Run your first scrape

### `scrape_game`

Use `scrape_game` when you have one or more specific game IDs. Important
parameters:

* `game_ids` – list of strings/ints. Leading zeros are optional.
* `data_format` – `'dataframe'` (default) returns a `pandas.DataFrame`; `'csv'`
  saves a CSV to disk.
* `data_dir` – directory for CSV output. Defaults to your home directory.

```python
# returns a pandas.DataFrame
nba_df = ns.scrape_game([21800001, 21800002])

# writes CSVs into the provided folder
ns.scrape_game([21800001, 21800002], data_format="csv", data_dir="/tmp/pbp")
```

Each scrape fetches the CDN play-by-play, box score and (if
`NBA_SCRAPER_BACKFILL_COORDS=1` is set) the shot chart in order to add spatial
coordinates for shot attempts.

### `scrape_from_files`

Use local files when you are working offline or wish to unit test against frozen
fixtures.

```python
# CDN fixtures: pass both play-by-play and box score
ns.scrape_from_files("playbyplay_0022400001.json", "boxscore_0022400001.json", kind="cdn_local")

# Legacy v2 dump: only one JSON file is required
ns.scrape_from_files("0021700001.json", kind="v2_local")
```

Under the hood the helper routes to `io_sources.parse_any`, so you can also call
that function directly when you need more control.

### `scrape_season`

Scrape every regular-season game for a specific season (four-digit year). The
`data_format` and `data_dir` arguments mirror `scrape_game`.

```python
# Build a dataframe that contains the entire 2019 season
nba_df = ns.scrape_season(2019)

# Persist CSV exports instead of keeping everything in memory
ns.scrape_season(2019, data_format="csv", data_dir="/tmp/nba-2019")
```

### `scrape_date_range`

Capture all **regular-season** games between two dates (inclusive). Dates must
be strings formatted as `YYYY-MM-DD`.

```python
nba_df = ns.scrape_date_range("2019-01-01", "2019-01-03")

ns.scrape_date_range(
    "2019-01-01",
    "2019-01-03",
    data_format="csv",
    data_dir="/tmp/nba-jan"
)
```

## Step 4 – Configure advanced options

* **Output control** – The default return type is a dataframe. Pass
  `data_format="csv"` to materialise files. Use `data_dir` to change where CSVs
  are written.
* **Lineup enrichment** – Lineups are automatically attached by default. If you
  only need raw events, call `io_sources.parse_any(..., mapping_yaml_path=None)`
  and skip `lineup_builder.attach_lineups` in your own workflow.
* **Event remapping** – Set the `NBA_SCRAPER_MAP` environment variable to point
  to a YAML file derived from `mapping_template.yml`. This lets you override
  turnover/foul/violation sub-types without editing code.
* **Shot coordinate backfill** – Export shot coordinates for every attempt by
  setting `NBA_SCRAPER_BACKFILL_COORDS=1`. The parser will fetch or load the
  matching shot chart and merge the coordinates into the returned dataframe.
* **Synthetic free-throw descriptions** – Legacy consumers can set
  `NBA_SCRAPER_SYNTH_FT_DESC=1` to emit “Free Throw N of M” text in the
  `homedescription` / `visitordescription` columns.

## Step 5 – Work with the canonical dataframe

The dataframe returned by any of the helpers is ready to feed into
[`nba_parser.PbP`](https://pypi.org/project/nba-parser/), a SQL warehouse or your
own analytics stack. Every row includes:

* A stable `eventnum` and `pctimestring`.
* Actor IDs plus the derived `event_team` and descriptive `event_type_de`.
* Lineup context (`home_player_1`…`home_player_5`, etc.) and score state.
* Structured flags for possessions, steals, blocks and free-throw trip counts.

Because CDN and v2 sources both map onto the same schema you can mix seasons and
data sources without any post-processing.

# Maintainer & contact

`nba_scraper` was founded by Matthew Barlowe, and the project will always be grateful for the groundwork he laid. Day-to-day maintenance now happens under Kostya Medvedovsky (@kmedved on Twitter, creator of DARKO).

If you have any troubles or bugs please **open an issue/bug report**. If you have
any improvements/suggestions please **submit a pull request**. For anything that
doesn't fit those buckets, please reach out to Kostya via GitHub issues or Twitter DMs (@kmedved).




