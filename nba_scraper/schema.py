"""Shared schema constants and helpers for canonical play-by-play rows."""
from __future__ import annotations

from typing import Any, List, Optional

EVENT_TYPE_DE = {
    1: "shot",
    2: "missed_shot",
    3: "free-throw",
    4: "rebound",
    5: "turnover",
    6: "foul",
    7: "violation",
    8: "substitution",
    9: "timeout",
    10: "jump-ball",
    12: "period",
    13: "period",
    15: "game",
}

# Canonical column order shared across CDN and v2 parsers.
CANONICAL_COLUMNS: List[str] = [
    "game_id",
    "period",
    "pctimestring",
    "seconds_elapsed",
    "event_length",
    "action_number",
    "order_number",
    "eventnum",
    "time_actual",
    "team_id",
    "team_tricode",
    "event_team",
    "player1_id",
    "player1_name",
    "player1_team_id",
    "player2_id",
    "player2_name",
    "player2_team_id",
    "player3_id",
    "player3_name",
    "player3_team_id",
    "home_team_id",
    "home_team_abbrev",
    "away_team_id",
    "away_team_abbrev",
    "homedescription",
    "visitordescription",
    "game_date",
    "season",
    "family",
    "subfamily",
    "eventmsgtype",
    "eventmsgactiontype",
    "event_type_de",
    "is_three",
    "shot_made",
    "points_made",
    "shot_distance",
    "x",
    "y",
    "side",
    "area",
    "area_detail",
    "assist_id",
    "block_id",
    "steal_id",
    "style_flags",
    "qualifiers",
    "is_o_rebound",
    "is_d_rebound",
    "team_rebound",
    "linked_shot_action_number",
    "possession_after",
    "score_home",
    "score_away",
    "scoremargin",
    "is_turnover",
    "is_steal",
    "is_block",
    "ft_n",
    "ft_m",
]


def int_or_zero(value: Any) -> int:
    try:
        if value in (None, ""):
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def scoremargin_str(score_home: Any, score_away: Any) -> str:
    try:
        if score_home in (None, "") or score_away in (None, ""):
            return ""
        return str(int(score_home) - int(score_away))
    except Exception:
        return ""


def points_made_from_family(family: str, shot_made: Optional[int]) -> int:
    if shot_made != 1:
        return 0
    if family == "3pt":
        return 3
    if family == "freethrow":
        return 1
    return 2


__all__ = [
    "CANONICAL_COLUMNS",
    "EVENT_TYPE_DE",
    "int_or_zero",
    "points_made_from_family",
    "scoremargin_str",
]
