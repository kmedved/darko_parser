# Changelog

## Unreleased

* Documented that Kostya Medvedovsky (@kmedved, creator of DARKO) now maintains `nba_scraper` and thanked Matthew Barlowe for creating the project.
* Switched to installing `nba_parser` directly from the upstream `kmedved/nba_parser` default branch instead of bundling a pinned copy.

## 1.2.2

* Normalised `game_date` from both CDN and v2 parsers to a consistent `YYYY-MM-DD` string.
* Fixed `teambygamestats()` to produce properly zero-padded `toc_string` values.
* Deferred RAPM imports so `nba_parser` can be used without scikit-learn and documented an optional `rapm` extra.
* Added regression tests covering the normalised dates and `toc_string` format.

## 1.2.1

* Added compatibility columns (`event_team`, `event_type_de`, player team IDs, turnover/steal/block flags, `ft_n`/`ft_m`, `season`) to both CDN and legacy v2 parsers for use with `nba_parser`.
* Populated lineup name columns (`home_player_1`…`away_player_5`) alongside existing ID fields in the lineup builder.
* Provided structured free-throw trip metadata and canonical season detection for all events.
* Bundled an updated `nba_parser` (0.2.2) with resilient scoremargin handling and structured free-throw possession detection.
* Added an integration smoke test to ensure the canonical dataframe round-trips through `nba_parser.PbP`.
* Documented the canonical schema and nba_parser workflow in the README.
