from __future__ import annotations
"""
Helpers for validating canonical PbP team totals against the official boxscore.

This module is intentionally lightweight and side-effect free so it can be used
both in tests and as an optional runtime sanity check.
"""

import logging
from typing import Any, Dict, List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


CORE_TEAM_STAT_FIELDS: Tuple[str, ...] = (
    "points",
    "fgm",
    "fga",
    "tpm",
    "tpa",
    "ftm",
    "fta",
)

EXTENDED_TEAM_STAT_FIELDS: Tuple[str, ...] = CORE_TEAM_STAT_FIELDS + (
    "rebounds",
    "assists",
    "steals",
    "blocks",
    "turnovers",
)


def _team_totals_from_pbp(df: pd.DataFrame) -> Dict[int, Dict[str, int]]:
    """
    Compute simple team-level totals directly from canonical PbP.

    We mirror the obvious box score stats:
      - points, FGM, FGA, 3PM, 3PA, FTM, FTA
      - rebounds, assists, steals, blocks, turnovers
    """
    if df.empty:
        return {}

    if "team_id" not in df or "family" not in df:
        return {}

    shots_mask = df["family"].isin(["2pt", "3pt"])
    ft_mask = df["family"] == "freethrow"
    shots = df[shots_mask]
    freethrows = df[ft_mask]

    totals: Dict[int, Dict[str, int]] = {}

    # Ensure we have a numeric view on points_made in case of unexpected dtypes.
    numeric_cache: Dict[str, pd.Series] = {}

    def _numeric_series(column: str) -> pd.Series:
        if column not in numeric_cache:
            if column in df:
                numeric_cache[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)
            else:
                numeric_cache[column] = pd.Series(0, index=df.index, dtype="float64")
        return numeric_cache[column]

    points_series = _numeric_series("points_made")
    shot_made_series = _numeric_series("shot_made")
    assist_series = _numeric_series("assist_id")
    o_reb_series = _numeric_series("is_o_rebound")
    d_reb_series = _numeric_series("is_d_rebound")
    team_reb_series = _numeric_series("team_rebound")
    turnover_series = _numeric_series("is_turnover")

    def _opponent_totals(value_col: str, team_col: str) -> Dict[int, int]:
        if value_col not in df or team_col not in df:
            return {}
        values = _numeric_series(value_col)
        teams = pd.to_numeric(df[team_col], errors="coerce").fillna(0)
        data = pd.DataFrame({"team": teams, "value": values})
        mask = (data["value"] > 0) & data["team"].notna() & (data["team"] != 0)
        grouped = data[mask].groupby("team")["value"].sum()
        result: Dict[int, int] = {}
        for tid, val in grouped.items():
            try:
                tid_int = int(tid)
            except (TypeError, ValueError):
                continue
            if tid_int:
                result[tid_int] = int(val)
        return result

    steals_by_team = _opponent_totals("is_steal", "player2_team_id")
    blocks_by_team = _opponent_totals("is_block", "player3_team_id")

    for team_id, group in df.groupby("team_id"):
        if not team_id:
            continue

        g_shots = shots[shots["team_id"] == team_id]
        g_fts = freethrows[freethrows["team_id"] == team_id]

        # Use the filtered group index to sum points from the global series,
        # so we benefit from the numeric coercion above.
        try:
            tid = int(team_id)
        except (TypeError, ValueError):
            continue

        pts = int(points_series.loc[group.index].sum())

        fga = int(len(g_shots))
        shot_made_values = shot_made_series.loc[g_shots.index]
        fgm = int((shot_made_values == 1).sum())
        shot_families = df["family"].loc[g_shots.index]
        tpa = int((shot_families == "3pt").sum())
        tpm = int(((shot_families == "3pt") & (shot_made_values == 1)).sum())
        fta = int(len(g_fts))
        ftm = int((shot_made_series.loc[g_fts.index] == 1).sum())

        made_shots = g_shots.loc[shot_made_values == 1]
        assists = 0
        if not made_shots.empty:
            assist_vals = assist_series.loc[made_shots.index]
            assists = int((assist_vals > 0).sum())

        rebounds = int(
            o_reb_series.loc[group.index].sum()
            + d_reb_series.loc[group.index].sum()
            + team_reb_series.loc[group.index].sum()
        )

        turnovers = int(turnover_series.loc[group.index].sum())

        totals[tid] = {
            "points": pts,
            "fgm": fgm,
            "fga": fga,
            "tpm": tpm,
            "tpa": tpa,
            "ftm": ftm,
            "fta": fta,
            "rebounds": rebounds,
            "assists": assists,
            "steals": int(steals_by_team.get(tid, 0)),
            "blocks": int(blocks_by_team.get(tid, 0)),
            "turnovers": turnovers,
        }

    return totals


def _team_totals_from_box(box_json: Dict[str, Any]) -> Dict[int, Dict[str, int]]:
    """
    Extract the same core stats from a CDN boxscore payload.

    We keep this defensive because the CDN schema can shift.
    """
    game = (box_json or {}).get("game", {})
    totals: Dict[int, Dict[str, int]] = {}

    def _side(key: str) -> None:
        team = game.get(key) or {}
        stats = team.get("statistics") or team.get("stats") or {}
        team_id = team.get("teamId") or stats.get("teamId")
        if not team_id:
            return

        def _get(*names: str) -> int:
            for name in names:
                if name in stats and stats[name] is not None:
                    try:
                        return int(stats[name])
                    except (TypeError, ValueError):
                        continue
            return 0

        # points: prefer stats["points"], fall back to scoreboard "score"
        points = _get("points")
        if not points and "score" in team and team["score"] is not None:
            try:
                points = int(team["score"])
            except (TypeError, ValueError):
                points = 0

        try:
            tid = int(team_id)
        except (TypeError, ValueError):
            return

        rebounds = _get("reboundsTotal", "totalRebounds", "rebounds")
        if not rebounds:
            rebounds = _get("reboundsOffensive", "offensiveRebounds") + _get(
                "reboundsDefensive", "defensiveRebounds"
            )

        totals[tid] = {
            "points": points,
            "fgm": _get("fieldGoalsMade", "fgm"),
            "fga": _get("fieldGoalsAttempted", "fga"),
            "tpm": _get("threePointersMade", "threePointsMade", "tpm"),
            "tpa": _get("threePointersAttempted", "threePointsAttempted", "tpa"),
            "ftm": _get("freeThrowsMade", "ftm"),
            "fta": _get("freeThrowsAttempted", "fta"),
            "rebounds": rebounds,
            "assists": _get("assists", "ast"),
            "steals": _get("steals", "stl"),
            "blocks": _get("blocks", "blk"),
            "turnovers": _get("turnovers", "to"),
        }

    for side_key in ("homeTeam", "awayTeam"):
        _side(side_key)

    return totals


def compare_pbp_to_box(
    df: pd.DataFrame,
    box_json: Dict[str, Any],
    *,
    fields: Tuple[str, ...] = CORE_TEAM_STAT_FIELDS,
    atol: int = 0,
) -> List[Tuple[int, str, int, int]]:
    """
    Compare PbP-derived team totals with the official boxscore.

    Returns a list of mismatches:
        [(team_id, field, pbp_value, box_value), ...]
    """
    pbp_totals = _team_totals_from_pbp(df)
    box_totals = _team_totals_from_box(box_json)
    mismatches: List[Tuple[int, str, int, int]] = []

    for team_id, pbp_vals in pbp_totals.items():
        box_vals = box_totals.get(team_id)
        if not box_vals:
            continue
        for field in fields:
            p = int(pbp_vals.get(field, 0))
            b_raw = box_vals.get(field)
            if b_raw is None:
                continue
            try:
                b = int(b_raw)
            except (TypeError, ValueError):
                continue
            if abs(p - b) > atol:
                mismatches.append((team_id, field, p, b))

    return mismatches


def log_team_boxscore_mismatches(
    df: pd.DataFrame,
    box_json: Dict[str, Any],
    *,
    atol: int = 0,
) -> None:
    """
    Convenience: compute mismatches and log them as warnings.
    """
    mismatches = compare_pbp_to_box(df, box_json, atol=atol)
    if not mismatches:
        return

    game_ids = df.get("game_id")
    if game_ids is not None:
        unique_ids = sorted(set(str(g) for g in game_ids.unique()))
        game_label = unique_ids[0] if len(unique_ids) == 1 else unique_ids
    else:
        game_label = "unknown"

    logger.warning("PbP/boxscore mismatches detected for game_id=%s", game_label)
    for team_id, field, pbp_val, box_val in mismatches:
        logger.warning(
            "  team_id=%s field=%s pbp=%s box=%s",
            team_id,
            field,
            pbp_val,
            box_val,
        )


__all__ = [
    "compare_pbp_to_box",
    "log_team_boxscore_mismatches",
    "CORE_TEAM_STAT_FIELDS",
    "EXTENDED_TEAM_STAT_FIELDS",
]
