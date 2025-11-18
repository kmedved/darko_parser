"""Helper utilities for the nba_scraper package."""
from __future__ import annotations

import datetime as dt
from typing import List

from . import cdn_client

SECONDS_PER_PERIOD = 720
SECONDS_PER_OT = 300


def iso_clock_to_pctimestring(iso_clock: str | None) -> str:
    if not iso_clock:
        return "12:00"
    iso_clock = iso_clock.strip().upper()
    if not iso_clock.startswith("PT"):
        return iso_clock
    minutes = 12
    seconds = 0
    if "M" in iso_clock:
        try:
            minutes_part = iso_clock.split("PT")[1].split("M")[0]
            minutes = int(float(minutes_part))
        except (IndexError, ValueError):
            minutes = 0
    if "S" in iso_clock:
        try:
            seconds_part = iso_clock.split("M")[-1].split("S")[0]
            seconds = int(float(seconds_part))
        except (IndexError, ValueError):
            seconds = 0
    return f"{minutes:02d}:{seconds:02d}"


def _period_base_seconds(period: int) -> int:
    if period <= 4:
        return SECONDS_PER_PERIOD * (period - 1)
    return SECONDS_PER_PERIOD * 4 + SECONDS_PER_OT * (period - 5)


def seconds_elapsed(period: int, pctimestring: str) -> int:
    if period is None:
        return 0
    try:
        minutes, seconds = map(int, pctimestring.split(":"))
    except Exception:
        minutes, seconds = 0, 0
    total_in_period = SECONDS_PER_PERIOD if period <= 4 else SECONDS_PER_OT
    remaining = minutes * 60 + seconds
    elapsed_in_period = total_in_period - remaining
    return _period_base_seconds(int(period)) + elapsed_in_period


def get_date_games(from_date: str, to_date: str) -> List[str]:
    start = dt.datetime.strptime(from_date, "%Y-%m-%d").date()
    end = dt.datetime.strptime(to_date, "%Y-%m-%d").date()
    if start.year < 2019:
        raise ValueError(
            "CDN schedule only supports 2019+ seasons; use scrape_from_files for older games."
        )

    schedule = cdn_client.fetch_schedule()
    game_dates = schedule.get("leagueSchedule", {}).get("gameDates", [])
    game_ids: List[str] = []
    for entry in game_dates:
        date_str = entry.get("gameDate")
        if not date_str:
            continue
        try:
            entry_date = dt.datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            continue
        if start <= entry_date <= end:
            for game in entry.get("games", []):
                gid = game.get("gameId") or game.get("gameID")
                if gid:
                    game_ids.append(str(gid))
    return game_ids


def get_season(date: dt.datetime) -> int:
    year = date.year
    if date.month < 7:
        return year - 1
    return year
