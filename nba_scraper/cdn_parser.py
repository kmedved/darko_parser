"""Parser that converts CDN liveData play-by-play feeds into canonical rows."""
from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from .mapping.descriptor_norm import canon_str, normalize_descriptor
from .mapping.event_codebook import (
    actiontype_code_for,
    eventmsgtype_for,
    ft_n_m,
)
from .helper_functions import get_season, iso_clock_to_pctimestring, seconds_elapsed
from .mapping.loader import load_mapping
from .parser_utils import _fill_team_fields, _synth_xy, finalize_dataframe
from .schema import (
    CANONICAL_COLUMNS,
    EVENT_TYPE_DE,
    int_or_zero,
    points_made_from_family,
    scoremargin_str,
)

_SYNTH_FT_DESC = os.getenv("NBA_SCRAPER_SYNTH_FT_DESC", "0") == "1"

def _resolve_player_name(*values: Any) -> str:
    """Return the first non-empty string from the provided values."""

    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            candidate = value.strip()
        else:
            candidate = str(value)
        if candidate:
            return candidate
    return ""

def _opponent_team_id(
    acting_team_id: int, home_team_id: Optional[int], away_team_id: Optional[int]
) -> int:
    if acting_team_id and home_team_id and acting_team_id == int(home_team_id):
        return int_or_zero(away_team_id)
    if acting_team_id and away_team_id and acting_team_id == int(away_team_id):
        return int_or_zero(home_team_id)
    return 0


def _jumpball_recovered_person(action: Dict[str, Any]) -> int:
    """
    Return the recovered player's id for a jump-ball action.

    CDN feeds aren't fully consistent:
      - many games use jumpBallRecoveredPersonId / jumpBallRecoverdPersonId
      - some 'subType == "recovered"' rows only expose the recovered
        player via personId or tertiaryPersonId

    We normalise all of those possibilities into a single integer id.
    """
    subtype = (action.get("subType") or "").strip().lower()

    candidates: List[Any] = [
        action.get("jumpBallRecoverdPersonId"),
        action.get("jumpBallRecoveredPersonId"),
    ]

    # On 'recovered' rows, fall back to the generic ids if the specialised
    # jumpBallRecovered* fields aren't populated.
    if subtype == "recovered":
        # Prefer tertiary ids when present since feeds often keep the jumpers
        # as primary/secondary person ids on the recovery row.
        candidates.append(action.get("tertiaryPersonId"))
        candidates.append(action.get("personId"))

    for value in candidates:
        if value not in (None, "", 0):
            return int_or_zero(value)
    return 0


class _SidecarCollector:
    def __init__(self) -> None:
        self._by_shot: Dict[Tuple[int, Optional[int]], Dict[str, Any]] = {}
        self._pending_rows: Dict[Tuple[int, Optional[int]], List[Dict[str, Any]]] = {}

    def register(self, action: Dict[str, Any]) -> None:
        shot_key = action.get("shotActionNumber")
        # Only treat block/steal sidecars that explicitly reference a shot.
        if shot_key is None:
            return
        key = (action.get("period"), shot_key)
        entry = self._by_shot.setdefault(key, {})
        if action.get("actionType") == "block":
            entry["block_id"] = action.get("personId")
            entry["block_name"] = action.get("playerName")
        if action.get("actionType") == "steal":
            entry["steal_id"] = action.get("personId")
            entry["steal_name"] = action.get("playerName")
        if key in self._pending_rows:
            for row in self._pending_rows.pop(key):
                self._apply_entry(entry, row)

    def apply(self, action: Dict[str, Any], row: Dict[str, Any]) -> None:
        shot_key = action.get("shotActionNumber")
        if shot_key is None:
            # No shotActionNumber means we cannot link to a sidecar entry.
            return
        key = (action.get("period"), shot_key)
        data = self._by_shot.get(key)
        if not data:
            self._pending_rows.setdefault(key, []).append(row)
            return
        self._apply_entry(data, row)

    def _apply_entry(self, data: Dict[str, Any], row: Dict[str, Any]) -> None:
        if data.get("block_id"):
            row["block_id"] = data["block_id"]
            if data.get("block_name"):
                row["player3_name"] = data["block_name"]
            if not int_or_zero(row.get("player3_team_id")):
                row["player3_team_id"] = _opponent_team_id(
                    int_or_zero(row.get("player1_team_id")),
                    row.get("home_team_id"),
                    row.get("away_team_id"),
                )
            if not int_or_zero(row.get("player3_id")):
                row["player3_id"] = data["block_id"]
            if row.get("family") in {"2pt", "3pt"} and row.get("shot_made") == 0:
                row["is_block"] = 1
        if data.get("steal_id"):
            row["steal_id"] = data["steal_id"]
            if data.get("steal_name"):
                row["player2_name"] = data["steal_name"]
            if row.get("family") == "turnover":
                row["player2_id"] = data["steal_id"]
                if not int_or_zero(row.get("player2_team_id")):
                    row["player2_team_id"] = _opponent_team_id(
                        int_or_zero(row.get("player1_team_id")),
                        row.get("home_team_id"),
                        row.get("away_team_id"),
                    )
                row["is_steal"] = 1


def _qualifiers_list(action: Dict[str, Any]) -> List[str]:
    quals = action.get("qualifiers") or []
    normalized = {canon_str(q) for q in quals if q}
    return sorted(q for q in normalized if q)


def _family_from_action(action: Dict[str, Any]) -> str:
    fam = (action.get("actionType") or "").lower()
    if fam in {"made shot", "made"}:
        fam = "2pt"
    return fam


def _shot_made_value(family: str, shot_result: Optional[str]) -> Optional[int]:
    if family not in {"2pt", "3pt", "freethrow"}:
        return None
    if shot_result is None:
        return None
    if shot_result.lower() == "made":
        return 1
    if shot_result.lower() == "missed":
        return 0
    return None


def _score_tuple(action: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    """
    Return (score_home, score_away) as integers or (None, None) if unavailable.

    Supports both:
      - legacy: action["score"] = {"home": int, "away": int}
      - new:    action["scoreHome"], action["scoreAway"]
    """
    # Legacy nested score dict
    score = action.get("score")
    if isinstance(score, dict):
        home_raw = score.get("home")
        away_raw = score.get("away")
    else:
        # New CDN fields are strings like "3"
        home_raw = action.get("scoreHome")
        away_raw = action.get("scoreAway")

    def _maybe_int(val: Any) -> Optional[int]:
        if val in (None, "", " "):
            return None
        try:
            return int(val)
        except (TypeError, ValueError):
            return None

    return _maybe_int(home_raw), _maybe_int(away_raw)


def _team_meta(box_json: Dict[str, Any]) -> Tuple[int, str, int, str, str]:
    game_meta = box_json.get("game", {})
    home = game_meta.get("homeTeam", {})
    away = game_meta.get("awayTeam", {})
    home_id = home.get("teamId")
    away_id = away.get("teamId")
    home_tri = home.get("teamTricode")
    away_tri = away.get("teamTricode")
    game_date = game_meta.get("gameTimeUTC")
    return home_id, home_tri, away_id, away_tri, game_date


def parse_actions_to_rows(
    pbp_json: Dict[str, Any],
    box_json: Dict[str, Any],
    mapping_yaml_path: Optional[str] = None,
) -> pd.DataFrame:
    actions = pbp_json.get("game", {}).get("actions", [])
    game_id = pbp_json.get("game", {}).get("gameId")
    home_id, home_tri, away_id, away_tri, game_date = _team_meta(box_json)
    # Build personId -> team mappings for fallback team attribution.
    person_to_team: Dict[int, int] = {}
    person_to_tri: Dict[int, str] = {}
    game_meta = (box_json or {}).get("game", {})
    for side in ("homeTeam", "awayTeam"):
        team_blob = game_meta.get(side) or {}
        team_id_val = int_or_zero(team_blob.get("teamId"))
        team_tri_val = team_blob.get("teamTricode") or ""
        for player in team_blob.get("players", []) or []:
            pid = int_or_zero(player.get("personId"))
            if not pid:
                continue
            person_to_team[pid] = team_id_val
            if team_tri_val:
                person_to_tri[pid] = team_tri_val
    game_timestamp = pd.to_datetime(game_date, utc=True, errors="coerce")
    season_val: Optional[int] = None
    if game_timestamp is not None and not pd.isna(game_timestamp):
        ts = game_timestamp
        if ts.tzinfo is not None:
            ts = ts.tz_convert(None)
        season_val = get_season(ts.to_pydatetime())

    mapping = load_mapping(mapping_yaml_path or os.getenv("NBA_SCRAPER_MAP"))

    rows: List[Dict[str, Any]] = []
    sidecars = _SidecarCollector()

    for action in actions:
        family = _family_from_action(action)
        if family in {"block", "steal"}:
            sidecars.register(action)
            continue

        pctimestring = iso_clock_to_pctimestring(action.get("clock"))
        secs = seconds_elapsed(action.get("period"), pctimestring)

        descriptor_core, style_flags = normalize_descriptor(action.get("descriptor"))
        subfamily_raw = action.get("subType") or descriptor_core
        subfamily_norm = canon_str(subfamily_raw)
        subfamily = subfamily_norm or subfamily_raw

        # Special handling for certain foul subtypes that are encoded via descriptor.
        # CDN often uses subType="personal" and puts the real type in `descriptor`.
        if family == "foul":
            st_norm = canon_str(action.get("subType") or "")
            desc_norm = canon_str(descriptor_core or "")
            # Shooting foul: "shooting personal FOUL"
            if desc_norm == "shooting" and st_norm == "personal":
                subfamily = "shooting"
            # Loose-ball foul: "loose ball personal FOUL"
            elif desc_norm == "loose ball" and (st_norm == "personal" or not st_norm):
                subfamily = "loose ball"

        flags = set(style_flags or [])
        if descriptor_core:
            d_norm = canon_str(descriptor_core)
            if d_norm and d_norm != canon_str(subfamily):
                flags.add(d_norm)
        style_flags = sorted(flags)
        shot_result = action.get("shotResult")
        eventmsgtype = eventmsgtype_for(family, shot_result, subfamily)
        eventmsgactiontype = actiontype_code_for(family, subfamily)
        shot_made = _shot_made_value(family, shot_result)
        points = points_made_from_family(family, shot_made)
        is_three = 1 if family == "3pt" else 0
        score_home, score_away = _score_tuple(action)

        team_id_raw = action.get("teamId")
        team_id_int = int_or_zero(team_id_raw)
        primary_player_id = int_or_zero(action.get("personId"))
        if not team_id_int and primary_player_id:
            team_id_int = person_to_team.get(primary_player_id, 0)
        event_team = action.get("teamTricode") or ""
        if not event_team:
            if team_id_int == int_or_zero(home_id):
                event_team = home_tri or person_to_tri.get(primary_player_id, "")
            elif team_id_int == int_or_zero(away_id):
                event_team = away_tri or person_to_tri.get(primary_player_id, "")
            elif primary_player_id:
                event_team = person_to_tri.get(primary_player_id, "")
        opp_team_id = _opponent_team_id(team_id_int, home_id, away_id)

        # Some newer CDN feeds attach block info directly to the shot action
        # instead of (or in addition to) a separate "block" sidecar row.
        block_pid_direct = int_or_zero(
            action.get("blockPersonId") or action.get("blockPersonID")
        )
        block_name_direct = (
            action.get("blockPlayerName")
            or action.get("blockPlayerNameInitial")
            or ""
        )

        qualifiers_list = _qualifiers_list(action)

        # Drop 'startperiod' on the opening jump ball so the qualifier is
        # reserved for the explicit period-start action, matching legacy v2
        # semantics.
        if (
            family == "jumpball"
            and action.get("period") == 1
            and secs == 0
            and "startperiod" in qualifiers_list
        ):
            qualifiers_list = [q for q in qualifiers_list if q != "startperiod"]

        style_flags = [
            flag
            for flag in (style_flags or [])
            if canon_str(flag) not in {"startperiod", "challenge"}
        ]

        sig_key = (
            canon_str(family),
            canon_str(subfamily),
            canon_str(descriptor_core),
            tuple(sorted(canon_str(q) for q in qualifiers_list)),
        )
        overrides = mapping.get(sig_key)

        ft_n_val: Optional[int] = None
        ft_m_val: Optional[int] = None
        if family == "freethrow":
            ft_n_val, ft_m_val = ft_n_m(action.get("subType") or "")
            # leave as None when unknown; possession inference must not guess

        row: Dict[str, Any] = {
            "game_id": game_id,
            "period": action.get("period"),
            "pctimestring": pctimestring,
            "seconds_elapsed": secs,
            "event_length": None,
            "action_number": action.get("actionNumber"),
            "order_number": action.get("orderNumber"),
            "eventnum": action.get("actionNumber"),
            "time_actual": action.get("timeActual"),
            "team_id": team_id_int,
            "team_tricode": event_team,
            "event_team": event_team,
            "player1_id": action.get("personId"),
            "player1_name": action.get("playerName"),
            "player1_team_id": team_id_int
            or person_to_team.get(primary_player_id, 0),
            "player2_id": action.get("secondaryPersonId"),
            "player2_name": action.get("secondaryPlayerName"),
            "player2_team_id": 0,
            "player3_id": action.get("tertiaryPersonId"),
            "player3_name": action.get("tertiaryPlayerName"),
            "player3_team_id": 0,
            "home_team_id": home_id,
            "home_team_abbrev": home_tri,
            "away_team_id": away_id,
            "away_team_abbrev": away_tri,
            "homedescription": "",
            "visitordescription": "",
            "game_date": game_timestamp.strftime("%Y-%m-%d") if pd.notna(game_timestamp) else None,
            "season": season_val if season_val is not None else 0,
            "family": family,
            "subfamily": subfamily,
            "eventmsgtype": eventmsgtype,
            "eventmsgactiontype": eventmsgactiontype,
            "event_type_de": EVENT_TYPE_DE.get(eventmsgtype, ""),
            "is_three": is_three,
            "shot_made": shot_made,
            "points_made": points,
            "shot_distance": action.get("shotDistance"),
            "x": action.get("x"),
            "y": action.get("y"),
            "side": action.get("side"),
            "area": action.get("area"),
            "area_detail": action.get("areaDetail"),
            "assist_id": action.get("assistPersonId"),
            "block_id": None,
            "steal_id": action.get("stealPersonId"),
            "style_flags": style_flags,
            "qualifiers": qualifiers_list,
            "is_o_rebound": 1 if family == "rebound" and subfamily == "offensive" else 0,
            "is_d_rebound": 1 if family == "rebound" and subfamily == "defensive" else 0,
            # Only mark team rebounds on actual rebound events with no playerId.
            "team_rebound": 1
            if (family == "rebound" and action.get("personId") in (0, None))
            else 0,
            "linked_shot_action_number": action.get("shotActionNumber"),
            "possession_after": action.get("possession"),
            "score_home": score_home,
            "score_away": score_away,
            "scoremargin": scoremargin_str(score_home, score_away),
            "is_turnover": 1 if family == "turnover" else 0,
            "is_steal": 0,
            "is_block": 0,
            "ft_n": ft_n_val,
            "ft_m": ft_m_val,
        }

        # For jump balls, prefer to follow v2 semantics:
        #   player1_id = jumpBallWonPlayer (home/away),
        #   player2_id = jumpBallLostPlayer,
        #   player3_id = jumpBallRecovered (always recorded).
        if family == "jumpball":
            won_id = int_or_zero(action.get("jumpBallWonPersonId"))
            lost_id = int_or_zero(action.get("jumpBallLostPersonId"))
            explicit_rec = (
                action.get("jumpBallRecoverdPersonId")
                or action.get("jumpBallRecoveredPersonId")
            )
            rec_id = _jumpball_recovered_person(action)
            # Determine teams for jumpers from person_to_team map.
            won_team = person_to_team.get(won_id, 0)
            lost_team = person_to_team.get(lost_id, 0)

            if won_id:
                row["player1_id"] = won_id
                row["player1_team_id"] = won_team
                row["player1_name"] = _resolve_player_name(
                    action.get("jumpBallWonPlayerName"), row.get("player1_name")
                )
            if lost_id:
                row["player2_id"] = lost_id
                row["player2_team_id"] = lost_team
                row["player2_name"] = _resolve_player_name(
                    action.get("jumpBallLostPlayerName"), row.get("player2_name")
                )
            # Always record the recovered player, even if they are also the
            # jumper who won/lost the tip. v2 feeds kept the recipient separate
            # and downstream logic can rely on player3_id being populated.
            if rec_id:
                # Preserve legacy behavior when the explicit jumpBallRecovered*
                # field is present: always override player3_id.
                if explicit_rec:
                    row["player3_id"] = rec_id
                else:
                    # Fallback path: only fill player3_id when we don't already
                    # have a non-zero tertiary id (e.g. from tertiaryPersonId).
                    if not int_or_zero(row.get("player3_id")):
                        row["player3_id"] = rec_id

                row["player3_team_id"] = person_to_team.get(rec_id, 0)
                row["player3_name"] = _resolve_player_name(
                    action.get("jumpBallRecoveredName"),
                    row.get("player3_name"),
                    action.get("playerName"),
                )

        if overrides:
            if family not in {"2pt", "3pt"} and "eventmsgtype" in overrides:
                row["eventmsgtype"] = int(overrides["eventmsgtype"])
                if "eventmsgactiontype" in overrides:
                    row["eventmsgactiontype"] = int(overrides["eventmsgactiontype"])
            if overrides.get("subfamily"):
                row["subfamily"] = str(overrides["subfamily"])
            row["event_type_de"] = EVENT_TYPE_DE.get(row["eventmsgtype"], "")

        sidecars.apply(action, row)

        # Map the CDN `description` into home/visitor columns for compatibility
        # with the v2-style schema. We don't attempt to fully parse who is home vs
        # away here; we simply assign the description to the acting team.
        desc = action.get("description") or ""
        if desc:
            if row.get("team_id") == home_id:
                row["homedescription"] = desc
                row["visitordescription"] = row.get("visitordescription", "")
            elif row.get("team_id") == away_id:
                row["visitordescription"] = desc
                row["homedescription"] = row.get("homedescription", "")
            else:
                # Fallback: put neutral events (period, jumpball, etc.) on home side.
                if not row.get("homedescription"):
                    row["homedescription"] = desc

        row["assist_id"] = int_or_zero(row.get("assist_id"))
        row["block_id"] = int_or_zero(row.get("block_id"))
        row["steal_id"] = int_or_zero(row.get("steal_id"))

        # If sidecars did not populate a block, fall back to the direct block fields
        if family in {"2pt", "3pt"} and shot_made == 0:
            if not row["block_id"] and block_pid_direct:
                row["block_id"] = block_pid_direct
                # Use block as player3 (blocker)
                if not int_or_zero(row.get("player3_id")):
                    row["player3_id"] = block_pid_direct
                if not int_or_zero(row.get("player3_team_id")):
                    # Blocker is on the opponent team
                    row["player3_team_id"] = _opponent_team_id(
                        team_id_int, home_id, away_id
                    )
                if not row.get("player3_name"):
                    row["player3_name"] = block_name_direct

        if family in {"2pt", "3pt"} and shot_made == 1:
            assist_id = row.get("assist_id")
            if assist_id:
                row["player2_id"] = assist_id
                row["player2_team_id"] = team_id_int
                row["player2_name"] = (
                    action.get("assistPlayerNameInitial")
                    or action.get("assistPlayerName")
                    or row.get("player2_name")
                    or ""
                )
        if family == "turnover":
            steal_id = row.get("steal_id")
            if steal_id:
                row["player2_id"] = steal_id
                row["player2_team_id"] = opp_team_id
                row["player2_name"] = (
                    action.get("stealPlayerName")
                    or row.get("player2_name")
                    or ""
                )
        if family == "foul":
            drawn_id = int_or_zero(action.get("foulDrawnPersonId"))
            if drawn_id:
                row["player2_id"] = drawn_id
                row["player2_team_id"] = person_to_team.get(drawn_id, 0)
                row["player2_name"] = (
                    action.get("foulDrawnPlayerName")
                    or row.get("player2_name")
                    or ""
                )
        if family in {"2pt", "3pt"} and shot_made == 0:
            block_id = row.get("block_id")
            if block_id:
                row["player3_id"] = block_id
                row["player3_team_id"] = opp_team_id
                row["player3_name"] = (
                    action.get("blockPlayerName")
                    or row.get("player3_name")
                    or ""
                )

        row["player1_team_id"] = int_or_zero(row.get("player1_team_id"))
        row["player2_id"] = int_or_zero(row.get("player2_id"))
        row["player3_id"] = int_or_zero(row.get("player3_id"))
        row["player2_team_id"] = int_or_zero(row.get("player2_team_id"))
        row["player3_team_id"] = int_or_zero(row.get("player3_team_id"))
        if row["player1_team_id"] == 0 and row["player1_id"]:
            row["player1_team_id"] = person_to_team.get(row["player1_id"], 0)
        if row["player2_team_id"] == 0 and row["player2_id"]:
            row["player2_team_id"] = person_to_team.get(row["player2_id"], 0)
        if row["player3_team_id"] == 0 and row["player3_id"]:
            row["player3_team_id"] = person_to_team.get(row["player3_id"], 0)
        eventmsgtype_final = int_or_zero(row.get("eventmsgtype"))
        row["event_type_de"] = EVENT_TYPE_DE.get(eventmsgtype_final, "")
        row["is_turnover"] = 1 if eventmsgtype_final == 5 else 0
        row["is_steal"] = (
            1 if row["is_turnover"] and int_or_zero(row.get("steal_id")) else 0
        )
        row["is_block"] = (
            1
            if family in {"2pt", "3pt"}
            and shot_made == 0
            and int_or_zero(row.get("block_id"))
            else 0
        )

        if (
            _SYNTH_FT_DESC
            and family == "freethrow"
            and ft_n_val is not None
            and ft_m_val is not None
        ):
            if row.get("team_id") == home_id:
                row["homedescription"] = f"Free Throw {ft_n_val} of {ft_m_val}"
                row["visitordescription"] = ""
            elif row.get("team_id") == away_id:
                row["visitordescription"] = f"Free Throw {ft_n_val} of {ft_m_val}"
                row["homedescription"] = ""

        _fill_team_fields(row)

        x_val = row.get("x")
        y_val = row.get("y")
        if family in {"2pt", "3pt"} and (x_val is None or y_val is None):
            synth_x, synth_y = _synth_xy(
                row.get("area"), row.get("area_detail"), row.get("side")
            )
            if synth_x is None or synth_y is None:
                fallback = {"2pt": (50.0, 50.0), "3pt": (50.0, 82.0)}.get(family)
                if fallback:
                    synth_x, synth_y = fallback
            if synth_x is not None and synth_y is not None:
                row["x"], row["y"] = synth_x, synth_y
                flags = set(row.get("style_flags") or [])
                flags.add("xy_synth")
                row["style_flags"] = sorted(flags)

        rows.append(row)

    df = pd.DataFrame(rows, columns=CANONICAL_COLUMNS)
    df = finalize_dataframe(
        df,
        sort_keys=["period", "order_number", "action_number"],
    )
    return df
