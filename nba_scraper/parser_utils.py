"""Shared helpers for parser post-processing tasks."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_ABOVE: Dict[str, Tuple[float, float]] = {
    "Left": (25.0, 82.0),
    "Left Center": (35.0, 82.0),
    "Center": (50.0, 82.0),
    "Right Center": (65.0, 82.0),
    "Right": (75.0, 82.0),
}

_MID: Dict[str, Tuple[float, float]] = {
    "8-16 Left": (30.0, 55.0),
    "8-16 Center": (50.0, 55.0),
    "8-16 Right": (70.0, 55.0),
    "16-24 Left": (30.0, 68.0),
    "16-24 Center": (50.0, 68.0),
    "16-24 Right": (70.0, 68.0),
}

_PAINT: Dict[str, Tuple[float, float]] = {
    "0-8 Left": (45.0, 30.0),
    "0-8 Center": (50.0, 30.0),
    "0-8 Right": (55.0, 30.0),
}

_CORN: Dict[str, Tuple[float, float]] = {
    "Left": (8.0, 10.0),
    "Right": (92.0, 10.0),
}

_RA: Dict[str, Tuple[float, float]] = {"0-8 Center": (50.0, 22.0)}


def _fill_team_fields(row: Dict[str, Any]) -> None:
    """Ensure team identifiers are populated for team-anchored events."""

    if not row.get("team_id"):
        if row.get("player1_team_id"):
            row["team_id"] = row["player1_team_id"]
        elif row.get("event_team"):
            if row["event_team"] == row.get("home_team_abbrev"):
                row["team_id"] = row.get("home_team_id")
            elif row["event_team"] == row.get("away_team_abbrev"):
                row["team_id"] = row.get("away_team_id")

    if not row.get("team_tricode"):
        if row.get("team_id") == row.get("home_team_id"):
            row["team_tricode"] = row.get("home_team_abbrev")
        elif row.get("team_id") == row.get("away_team_id"):
            row["team_tricode"] = row.get("away_team_abbrev")

    if not row.get("event_team"):
        row["event_team"] = row.get("team_tricode") or ""


def _synth_xy(area: str, area_detail: str, side: str) -> tuple[Optional[float], Optional[float]]:
    area_lower = (area or "").lower()
    detail = area_detail or ""
    side_norm = (side or "").capitalize()
    if "restricted" in area_lower:
        return _RA.get("0-8 Center", (None, None))
    if "non-ra" in area_lower or "paint" in area_lower:
        return _PAINT.get(detail, (None, None))
    if "mid-range" in area_lower:
        return _MID.get(detail, (None, None))
    if "corner 3" in area_lower:
        return _CORN.get(side_norm, (None, None))
    if "above the break 3" in area_lower or "above the break" in area_lower:
        key = detail if detail else side_norm
        return _ABOVE.get(key, (None, None))
    return (None, None)


def infer_possession_after(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill in possession_after so it reflects who has the ball *after* each event.

    For CDN data, possession_after is the canonical "team that will have the ball
    after this event" marker, derived from either the feed's possession token
    (when provided) or the heuristics below. Downstream consumers (e.g.,
    nba_scraper.stats.PbP) will only derive home/away possession flags from this
    column when it contains at least one home_team_id or away_team_id value;
    otherwise they fall back to legacy v2 heuristics. This guard keeps
    possession flags populated when feeds omit usable possession owners.

    Heuristics:
      - Turnover (eventmsgtype 5): flip to opponent(team_id).
      - Made 2PT/3PT (family '2pt'/'3pt' and shot_made == 1): flip to opponent(team_id).
      - Defensive rebound (family 'rebound' and is_d_rebound == 1): team_id.
      - Last made FT of a trip (family 'freethrow', shot_made == 1, ft_n == ft_m): flip to opponent(team_id).
      - Otherwise: fall back to the raw feed's possession value (possession_after).
    Then we smoothly propagate within (game_id, period) over *live* events only
    (we do not smear possession onto 'period' or 'timeout' rows).
    """

    df = df.copy()

    def _safe_team(value: Any) -> int:
        if pd.isna(value):
            return 0
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    # Pre-compute home/away team IDs once per game (assumed constant).
    home_team = _safe_team(df["home_team_id"].iloc[0] if "home_team_id" in df.columns else 0)
    away_team = _safe_team(df["away_team_id"].iloc[0] if "away_team_id" in df.columns else 0)

    def _opponent(team_id: int) -> int:
        if team_id == 0 or home_team == 0 or away_team == 0:
            return 0
        return away_team if team_id == home_team else home_team

    # Start from whatever is currently in possession_after (raw feed values for CDN).
    poss_init = pd.to_numeric(df.get("possession_after"), errors="coerce").fillna(0).astype(int)

    def _infer_row(row: pd.Series) -> int:
        team_id = _safe_team(row.get("team_id"))
        event_type = _safe_team(row.get("eventmsgtype"))
        family = row.get("family")
        sub_text = str(row.get("subfamily") or "").lower()

        shot_made = _safe_team(row.get("shot_made"))
        ft_n_val = _safe_team(row.get("ft_n"))
        ft_m_val = _safe_team(row.get("ft_m"))
        is_d_reb = _safe_team(row.get("is_d_rebound"))
        is_o_reb = _safe_team(row.get("is_o_rebound"))
        team_reb = _safe_team(row.get("team_rebound"))
        subfamily_lower = str(row.get("subfamily") or "").lower()
        is_technical_ft = "technical" in subfamily_lower

        last_ft_flag = row.get("is_last_ft")
        has_last_ft_flag = last_ft_flag is not None and not pd.isna(last_ft_flag)
        is_last_ft_flagged = bool(_safe_team(last_ft_flag) == 1) if has_last_ft_flag else False

        # 1) Heuristic "hint" based on the event itself.
        hint = 0

        # Turnover -> opponent.
        if event_type == 5:  # turnover
            hint = _opponent(team_id)

        # Made 2PT / 3PT -> opponent.
        elif family in {"2pt", "3pt"} and shot_made == 1:
            hint = _opponent(team_id)

        # Any rebound -> rebounder's team. Team rebounds cover feeds that omit
        # the offensive/defensive flag, and offensive rebounds ensure v2-only
        # games keep possession with the offense.
        elif family == "rebound" and (is_d_reb == 1 or is_o_reb == 1 or team_reb == 1):
            hint = team_id

        # Last made FT in a trip -> opponent, unless the raw feed already
        # provided a possession hint (e.g., technical FT where offense keeps).
        elif (
            family == "freethrow"
            and ft_n_val != 0
            and ft_m_val != 0
            and ft_n_val == ft_m_val
            and shot_made == 1
            and not is_technical_ft
        ):
            technical_like = any(
                token in sub_text for token in ("technical", "clear path", "illegal defense")
            )
            raw_val = row.get("possession_after")
            if technical_like or (has_last_ft_flag and not is_last_ft_flagged):
                if pd.notna(raw_val) and raw_val not in ("", 0):
                    try:
                        return int(raw_val)
                    except (TypeError, ValueError):
                        return 0
                return 0

            if pd.isna(raw_val) or raw_val in ("", 0):
                hint = _opponent(team_id)

        if hint:
            return int(hint)

        # 2) Fall back to raw possession_after value if present.
        raw_val = row.get("possession_after")
        if pd.notna(raw_val) and raw_val not in ("", 0):
            try:
                return int(raw_val)
            except (TypeError, ValueError):
                return 0

        return 0

    hints = df.apply(_infer_row, axis=1)
    # Use hint when non-zero; otherwise keep the initial value.
    poss_new = pd.Series(
        np.where(hints != 0, hints.to_numpy(), poss_init.to_numpy()),
        index=df.index,
        dtype="Int64",
    )

    # Do not propagate onto period/timeout rows.
    live_mask = ~df["event_type_de"].isin(["period", "timeout"])

    poss_live = poss_new.where(live_mask)
    poss_live = poss_live.mask(poss_live == 0)
    poss_live = poss_live.groupby([df["game_id"], df["period"]]).ffill().bfill()
    poss_live = poss_live.fillna(0).astype("Int64")

    df["possession_after"] = poss_new.where(~live_mask, poss_live)
    return df


def finalize_dataframe(df: pd.DataFrame, *, sort_keys: List[str]) -> pd.DataFrame:
    """Common end-of-parser post-processing for canonical frames."""

    if df.empty:
        return df

    df = df.sort_values(sort_keys, kind="mergesort")
    df["event_length"] = df.groupby("period")["seconds_elapsed"].diff()
    df["event_length"] = df["event_length"].fillna(0).abs()
    df = infer_possession_after(df)
    return df.reset_index(drop=True)


__all__ = [
    "_fill_team_fields",
    "_synth_xy",
    "infer_possession_after",
    "finalize_dataframe",
]
