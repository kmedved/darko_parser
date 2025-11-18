"""Utility helpers that operate on the canonical dataframe."""
from __future__ import annotations

from typing import Any

import numpy as np


def made_shot(row: Any) -> float:
    value = row.get("shot_made")
    if value is None:
        return np.nan
    return float(value)


def parse_foul(row: Any) -> str:
    if row.get("eventmsgtype") != 6:
        return ""
    return row.get("subfamily") or ""


def parse_shot_types(row: Any) -> str:
    if row.get("family") in {"2pt", "3pt", "freethrow"}:
        return row.get("subfamily") or ""
    return ""


def create_seconds_elapsed(row: Any) -> int:
    return int(row.get("seconds_elapsed", 0))


def calc_points_made(row: Any) -> int:
    return int(row.get("points_made", 0))
