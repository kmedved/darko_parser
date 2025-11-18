"""Utilities for attaching on-court lineups to the canonical dataframe."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd


def _name_map_from_box_and_pbp(
    box_json: dict, pbp_json: Optional[dict] = None
) -> Dict[int, str]:
    m: Dict[int, str] = {}
    game_blob = (box_json or {}).get("game", {})

    def _iter_players() -> List[dict]:
        collected: List[dict] = []
        base_players = game_blob.get("players") or []
        if isinstance(base_players, list):
            collected.extend(base_players)
        for side in ("homeTeam", "awayTeam"):
            side_blob = game_blob.get(side) or {}
            team_players = side_blob.get("players") or []
            if isinstance(team_players, list):
                collected.extend(team_players)
        return collected

    for player in _iter_players():
        pid = player.get("personId")
        name = player.get("name") or player.get("firstNameLastName")
        if not name:
            first = (
                player.get("firstName")
                or player.get("first_name")
                or player.get("first")
            )
            last = (
                player.get("familyName")
                or player.get("lastName")
                or player.get("last_name")
            )
            parts = [part for part in (first, last) if part]
            name = " ".join(parts)
        if pid and name:
            m[int(pid)] = str(name)
    if pbp_json:
        for action in pbp_json.get("game", {}).get("actions", []):
            pid = action.get("personId")
            name = action.get("playerName")
            if pid and name and int(pid) not in m:
                m[int(pid)] = str(name)
    return m


def _extract_starters_from_box(box_json: dict) -> Dict[str, List[int]]:
    def side(key: str) -> List[int]:
        side_obj = (box_json or {}).get("game", {}).get(key, {})
        starters: List[int] = []
        for player in side_obj.get("players", []):
            pid = player.get("personId")
            if not pid:
                continue
            if player.get("starter") or player.get("starterPosition"):
                starters.append(int(pid))
        if len(starters) < 5:
            pool: List[int] = []
            for player in side_obj.get("players", []):
                pid = player.get("personId")
                if not pid:
                    continue
                if player.get("status") == "ACTIVE":
                    pool.append(int(pid))
            starters = (
                starters
                + [pid for pid in pool if pid not in starters]
            )[:5]
        return starters if len(starters) == 5 else []

    return {"home": side("homeTeam"), "away": side("awayTeam")}

_LINEUP_ID_COLUMNS = [
    "home_player_1_id",
    "home_player_2_id",
    "home_player_3_id",
    "home_player_4_id",
    "home_player_5_id",
    "away_player_1_id",
    "away_player_2_id",
    "away_player_3_id",
    "away_player_4_id",
    "away_player_5_id",
]

_LINEUP_NAME_COLUMNS = [
    "home_player_1",
    "home_player_2",
    "home_player_3",
    "home_player_4",
    "home_player_5",
    "away_player_1",
    "away_player_2",
    "away_player_3",
    "away_player_4",
    "away_player_5",
]


def _init_lineup() -> List[Optional[int]]:
    return [None] * 5


def _update_with_player(lineup: List[Optional[int]], player_id: Optional[int]) -> None:
    if not player_id:
        return
    if player_id in lineup:
        return
    for idx, value in enumerate(lineup):
        if value is None:
            lineup[idx] = player_id
            return


def _copy_lineup(lineup: List[Optional[int]]) -> List[Optional[int]]:
    return list(lineup)


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value in (None, ""):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def extract_starters_from_box(box_json: Dict[str, Any]) -> Dict[str, List[int]]:
    """Extract starter ids for each team from a CDN box score payload."""
    return _extract_starters_from_box(box_json)


def _seed_lineup(starters: List[int]) -> List[Optional[int]]:
    lineup = _init_lineup()
    for idx, pid in enumerate(starters[:5]):
        pid_int = _safe_int(pid)
        lineup[idx] = pid_int if pid_int else None
    return lineup


def _apply_substitution(
    lineup: List[Optional[int]],
    sub_out: Optional[int],
    sub_in: Optional[int],
) -> None:
    """Apply a substitution to the running lineup in-place."""
    if not lineup:
        return

    sub_out = sub_out or None
    sub_in = sub_in or None

    if not sub_out and not sub_in:
        return

    if sub_in and sub_in in lineup:
        # Treat duplicate "sub in" events as bookkeeping noise. Removing the
        # outgoing player here would leave a four-player lineup that later gets
        # column-wise backfilled into duplicated IDs.
        return

    if sub_out and sub_in:
        if sub_out in lineup:
            idx = lineup.index(sub_out)
            lineup[idx] = sub_in
        else:
            lineup[0] = sub_in
        return

    if sub_out and not sub_in:
        if sub_out in lineup:
            idx = lineup.index(sub_out)
            lineup[idx] = None
        return

    if sub_in and sub_in not in lineup:
        for idx, pid in enumerate(lineup):
            if pid is None:
                lineup[idx] = sub_in
                return
        # No available slot and no matching sub_out – we do not know who left
        # the floor, so leave the lineup unchanged.
        return


def attach_lineups(
    df: pd.DataFrame,
    starters: Optional[Dict[str, List[int]]] = None,
    box_json: Optional[Dict[str, Any]] = None,
    pbp_json: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    if df.empty:
        for column in _LINEUP_ID_COLUMNS + _LINEUP_NAME_COLUMNS:
            df[column] = []
        return df

    df = df.copy()

    if starters is None and box_json is not None:
        starters = _extract_starters_from_box(box_json)

    if (
        starters
        and len(starters.get("home", [])) == 5
        and len(starters.get("away", [])) == 5
    ):
        home_seed = [int(pid) for pid in starters["home"]]
        away_seed = [int(pid) for pid in starters["away"]]

        def _seed_group(group: pd.DataFrame) -> pd.DataFrame:
            group = group.copy()
            if group.empty:
                return group
            if "period" in group.columns:
                period_one = group.index[group["period"] == 1]
                first_idx = period_one[0] if len(period_one) else group.index[0]
            else:
                first_idx = group.index[0]
            for i, pid in enumerate(home_seed, start=1):
                group.loc[first_idx, f"home_player_{i}_id"] = int(pid)
            for i, pid in enumerate(away_seed, start=1):
                group.loc[first_idx, f"away_player_{i}_id"] = int(pid)
            return group

        df = df.groupby("game_id", group_keys=False).apply(_seed_group)

    home_id = _safe_int(df["home_team_id"].iloc[0])
    away_id = _safe_int(df["away_team_id"].iloc[0])

    home_starters = starters.get("home", []) if starters else []
    away_starters = starters.get("away", []) if starters else []

    home_lineup = _seed_lineup(home_starters) if len(home_starters) == 5 else _init_lineup()
    away_lineup = _seed_lineup(away_starters) if len(away_starters) == 5 else _init_lineup()

    home_candidates: List[int] = [pid for pid in home_lineup if pid]
    away_candidates: List[int] = [pid for pid in away_lineup if pid]

    home_history: List[List[Optional[int]]] = []
    away_history: List[List[Optional[int]]] = []

    # For CDN-style substitutions we see separate "out" and "in" events.
    # Maintain simple FIFO queues of pending outs for each team.
    pending_out_home: List[int] = []
    pending_out_away: List[int] = []
    pending_secs_home: Optional[int] = None
    pending_secs_away: Optional[int] = None

    def _resolve_team_from_row(row: pd.Series) -> Optional[int]:
        for key in ("team_id", "player1_team_id", "player2_team_id", "player3_team_id"):
            value = _safe_int(row.get(key))
            if value:
                return value
        return None

    for _, row in df.iterrows():
        secs_val = _safe_int(row.get("seconds_elapsed"))
        player_id = _safe_int(row.get("player1_id"))
        event_team_id = _resolve_team_from_row(row) or _safe_int(row.get("player1_team_id"))
        if event_team_id and home_id and event_team_id == home_id:
            if player_id and player_id not in home_candidates:
                home_candidates.append(player_id)
                _update_with_player(home_lineup, player_id)
        elif event_team_id and away_id and event_team_id == away_id:
            if player_id and player_id not in away_candidates:
                away_candidates.append(player_id)
                _update_with_player(away_lineup, player_id)

        if row.get("family") == "substitution":
            subfamily = (row.get("subfamily") or "").strip().lower()
            raw_player = _safe_int(row.get("player1_id"))

            # Decide which team's queue to use.
            substitution_team = event_team_id
            if not substitution_team:
                if _safe_int(row.get("player1_team_id")) == home_id:
                    substitution_team = home_id
                elif _safe_int(row.get("player1_team_id")) == away_id:
                    substitution_team = away_id

            if not substitution_team:
                # Try to infer from current lineups (useful for older v2-style data).
                if raw_player and raw_player in home_lineup:
                    substitution_team = home_id
                elif raw_player and raw_player in away_lineup:
                    substitution_team = away_id

            if substitution_team and home_id and substitution_team == home_id:
                pending_out = pending_out_home
                target = home_lineup
                candidates = home_candidates
            elif substitution_team and away_id and substitution_team == away_id:
                pending_out = pending_out_away
                target = away_lineup
                candidates = away_candidates
            else:
                pending_out = None
                target = None
                candidates = None

            sub_out: Optional[int] = None
            sub_in: Optional[int] = None

            if subfamily in {"out"}:
                # CDN: "substitution" + subType "out" – only outgoing player present.
                if pending_out is not None and raw_player:
                    if substitution_team == home_id:
                        if pending_secs_home is None or pending_secs_home != secs_val:
                            pending_out.clear()
                        pending_secs_home = secs_val
                    elif substitution_team == away_id:
                        if pending_secs_away is None or pending_secs_away != secs_val:
                            pending_out.clear()
                        pending_secs_away = secs_val
                    pending_out.append(raw_player)
            elif subfamily in {"in"}:
                # CDN: "substitution" + subType "in" – only incoming player present.
                sub_in = raw_player
                if pending_out is not None:
                    if substitution_team == home_id:
                        if pending_secs_home is None or pending_secs_home != secs_val:
                            pending_out.clear()
                        pending_secs_home = secs_val
                    elif substitution_team == away_id:
                        if pending_secs_away is None or pending_secs_away != secs_val:
                            pending_out.clear()
                        pending_secs_away = secs_val
                    if pending_out:
                        sub_out = pending_out.pop(0)
            else:
                # v2-style: player1_id = out, player2_id = in on the same row.
                sub_out = raw_player
                sub_in = _safe_int(row.get("player2_id"))

            if target is not None and (sub_out or sub_in):
                # Ensure the incoming player is tracked as a candidate.
                if sub_in and candidates is not None and sub_in not in candidates:
                    candidates.append(sub_in)

                _apply_substitution(target, sub_out, sub_in)

        home_history.append(_copy_lineup(home_lineup))
        away_history.append(_copy_lineup(away_lineup))

    for idx in range(5):
        df[f"home_player_{idx + 1}_id"] = [line[idx] for line in home_history]
        df[f"away_player_{idx + 1}_id"] = [line[idx] for line in away_history]

    id_cols = [
        f"{team}_player_{slot}_id"
        for team in ("home", "away")
        for slot in range(1, 6)
    ]
    df[id_cols] = (
        df.groupby("game_id", group_keys=False)[id_cols]
        .ffill()
        .bfill()
    )
    df[id_cols] = df[id_cols].fillna(0).astype("Int64")

    name_map: Dict[int, str] = {}
    for id_col, name_col in [
        ("player1_id", "player1_name"),
        ("player2_id", "player2_name"),
        ("player3_id", "player3_name"),
    ]:
        if id_col not in df.columns or name_col not in df.columns:
            continue
        for pid, name in zip(df[id_col], df[name_col]):
            try:
                pid_int = int(pid)
            except (TypeError, ValueError):
                continue
            if not name:
                continue
            name_map.setdefault(pid_int, str(name))

    for pid, name in _name_map_from_box_and_pbp(box_json or {}, pbp_json).items():
        name_map[pid] = name

    # Backfill missing player names using any IDs we already captured.
    for id_col, name_col in [
        ("player1_id", "player1_name"),
        ("player2_id", "player2_name"),
        ("player3_id", "player3_name"),
    ]:
        if id_col in df.columns and name_col in df.columns:
            mask = df[name_col].isna() | (df[name_col] == "")

            def _map_backfill(value: Any) -> str:
                if pd.isna(value):
                    return ""
                try:
                    pid_int = int(value)
                except (TypeError, ValueError):
                    return ""
                if pid_int in name_map:
                    return name_map.get(pid_int, "")
                return ""

            if mask.any():
                df.loc[mask, name_col] = df.loc[mask, id_col].map(_map_backfill)

    def _lookup_name(value: Any) -> str:
        if pd.isna(value):
            return ""
        try:
            pid_int = int(value)
        except (TypeError, ValueError):
            return ""
        if pid_int == 0:
            return ""
        return name_map.get(pid_int, "")

    for team in ("home", "away"):
        for slot in range(1, 6):
            id_col = f"{team}_player_{slot}_id"
            name_col = f"{team}_player_{slot}"
            if id_col in df.columns:
                df[name_col] = df[id_col].map(_lookup_name)

    return df
