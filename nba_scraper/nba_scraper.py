"""Public entry points for scraping NBA play-by-play data."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd

from . import helper_functions as hf
from . import io_sources, scrape_functions


def check_format(data_format: str) -> None:
    if data_format.lower() not in {"pandas", "csv"}:
        raise ValueError("data_format must be either 'pandas' or 'csv'")


def check_valid_dates(from_date: str, to_date: str) -> None:
    start = pd.Timestamp(from_date)
    end = pd.Timestamp(to_date)
    if end < start:
        raise ValueError("date_to must be on or after date_from")


def _concat_or_write(frames: Sequence[pd.DataFrame], data_format: str, data_dir: str | Path):
    if not frames:
        return None
    df = pd.concat(frames).reset_index(drop=True)
    if data_format == "pandas":
        return df
    Path(data_dir).expanduser().mkdir(parents=True, exist_ok=True)
    output_path = Path(data_dir) / "nba_pbp.csv"
    df.to_csv(output_path, index=False, encoding="utf-8")
    return None


def scrape_game(game_ids: Iterable[str], data_format: str = "pandas", data_dir: str | Path = Path.home()):
    check_format(data_format)
    mapping = os.getenv("NBA_SCRAPER_MAP")
    frames: List[pd.DataFrame] = []
    for gid in game_ids:
        frames.append(scrape_functions.main_scrape(str(gid), mapping))
    return _concat_or_write(frames, data_format, data_dir)


def scrape_date_range(
    date_from: str,
    date_to: str,
    data_format: str = "pandas",
    data_dir: str | Path = Path.home(),
):
    check_format(data_format)
    check_valid_dates(date_from, date_to)
    game_ids = hf.get_date_games(date_from, date_to)
    return scrape_game(game_ids, data_format=data_format, data_dir=data_dir)


def scrape_season(
    season: int,
    data_format: str = "pandas",
    data_dir: str | Path = Path.home(),
):
    start_year = season
    end_year = season + 1
    date_from = f"{start_year}-10-01"
    date_to = f"{end_year}-06-30"
    return scrape_date_range(date_from, date_to, data_format=data_format, data_dir=data_dir)


def scrape_from_files(
    pbp_path: str,
    box_path: str | None = None,
    kind: str = "cdn_local",
    data_format: str = "pandas",
    data_dir: str | Path = Path.home(),
):
    check_format(data_format)
    mapping = os.getenv("NBA_SCRAPER_MAP")
    source_kind = io_sources.SourceKind(kind)
    if source_kind == io_sources.SourceKind.CDN_LOCAL:
        if box_path is None:
            raise ValueError("cdn_local requires both pbp_path and box_path")
        df = io_sources.parse_any((pbp_path, box_path), source_kind, mapping)
    elif source_kind == io_sources.SourceKind.V2_LOCAL:
        df = io_sources.parse_any(pbp_path, source_kind, mapping)
    else:
        raise ValueError("scrape_from_files supports 'cdn_local' or 'v2_local'")

    if data_format == "pandas":
        return df
    Path(data_dir).expanduser().mkdir(parents=True, exist_ok=True)
    output_path = Path(data_dir) / (Path(pbp_path).stem + ".csv")
    # Write UTF-8 CSV to avoid mangling non-ASCII player names (e.g. "Jović").
    df.to_csv(output_path, index=False, encoding="utf-8")
    return None
