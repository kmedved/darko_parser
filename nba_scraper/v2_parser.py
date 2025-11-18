"""Parser that converts legacy stats.nba.com v2 JSON into canonical rows."""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .mapping.descriptor_norm import canon_str, normalize_descriptor
from .mapping.event_codebook import ft_n_m
from .helper_functions import get_season, seconds_elapsed
from .mapping.loader import load_mapping
from .parser_utils import _fill_team_fields, _synth_xy, finalize_dataframe
from .schema import (
    CANONICAL_COLUMNS,
    EVENT_TYPE_DE,
    int_or_zero,
    points_made_from_family,
    scoremargin_str,
)


def _load_dataframe(v2_json: Dict) -> pd.DataFrame:
    result_sets = v2_json.get("resultSets") or []
    if not result_sets:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    rs = result_sets[0]
    headers = [h.lower() for h in rs.get("headers", [])]
    rows = rs.get("rowSet", [])
    return pd.DataFrame(rows, columns=headers)


def _infer_team_meta(df: pd.DataFrame) -> Tuple[Optional[int], Optional[str], Optional[int], Optional[str]]:
    seen: List[Tuple[Optional[int], Optional[str]]] = []
    for col_id, col_abbrev in [
        ("player1_team_id", "player1_team_abbreviation"),
        ("player2_team_id", "player2_team_abbreviation"),
        ("player3_team_id", "player3_team_abbreviation"),
        ("team_id", "team_abbreviation"),
    ]:
        if col_id not in df or col_abbrev not in df:
            continue
        for tid, abbrev in zip(df[col_id], df[col_abbrev]):
            if pd.isna(tid) or pd.isna(abbrev):
                continue
            pair = (int(tid), str(abbrev))
            if pair not in seen:
                seen.append(pair)
            if len(seen) >= 2:
                break
        if len(seen) >= 2:
            break
    if len(seen) == 1:
        seen.append((None, None))
    away = seen[0] if seen else (None, None)
    home = seen[1] if len(seen) > 1 else (None, None)
    return home[0], home[1], away[0], away[1]


def _family(row: pd.Series) -> str:
    eventmsgtype = int(row.get("eventmsgtype", 0) or 0)
    if eventmsgtype in (1, 2):
        desc = "{} {}".format(row.get("homedescription", ""), row.get("visitordescription", ""))
        desc = desc.upper()
        if "3PT" in desc:
            return "3pt"
        return "2pt"
    mapping = {
        3: "freethrow",
        4: "rebound",
        5: "turnover",
        6: "foul",
        7: "violation",
        8: "substitution",
        9: "timeout",
        10: "jumpball",
        12: "period",
        13: "period",
        15: "game",
    }
    return mapping.get(eventmsgtype, "")


def _shot_made(row: pd.Series) -> Optional[int]:
    eventmsgtype = int(row.get("eventmsgtype", 0) or 0)
    desc = "{} {}".format(row.get("homedescription", ""), row.get("visitordescription", "")).upper()
    if eventmsgtype == 1:
        return 1
    if eventmsgtype == 2:
        return 0
    if eventmsgtype == 3:
        if "MISS" in desc:
            return 0
        if desc.strip():
            return 1
    return None


def _subfamily(row: pd.Series, family: str, descriptor_core: str) -> str:
    raw = row.get("eventmsgactiontype")
    if pd.isna(raw):
        raw = None
    if family in {"turnover", "foul", "violation"} and descriptor_core:
        return descriptor_core
    if raw is None:
        return descriptor_core
    return str(int(raw))


def _team_id_for_row(row: pd.Series) -> Optional[int]:
    for key in ("team_id", "player1_team_id", "player2_team_id", "player3_team_id"):
        value = row.get(key)
        if not pd.isna(value):
            return int(value)
    return None


def _team_tricode_for_row(row: pd.Series) -> Optional[str]:
    for key in (
        "team_abbreviation",
        "player1_team_abbreviation",
        "player2_team_abbreviation",
        "player3_team_abbreviation",
    ):
        value = row.get(key)
        if value and isinstance(value, str):
            return value
    return None


def _ft_trip_from_text(row: pd.Series) -> Tuple[Optional[int], Optional[int]]:
    text_parts: List[str] = []
    for key in ("homedescription", "visitordescription"):
        val = row.get(key)
        if isinstance(val, str):
            text_parts.append(val)
    if not text_parts:
        return None, None

    text = " ".join(text_parts)
    ft_n_val, ft_m_val = ft_n_m(text)
    if ft_n_val is None or ft_m_val is None:
        return None, None
    return ft_n_val, ft_m_val


def parse_v2_to_rows(v2_json: Dict, mapping_yaml_path: Optional[str] = None) -> pd.DataFrame:
    df_raw = _load_dataframe(v2_json)
    if df_raw.empty:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    home_id, home_tri, away_id, away_tri = _infer_team_meta(df_raw)
    mapping = load_mapping(mapping_yaml_path or os.getenv("NBA_SCRAPER_MAP"))

    rows: List[Dict[str, object]] = []
    for _, row in df_raw.iterrows():
        family = _family(row)
        pctimestring = row.get("pctimestring") or "12:00"
        secs = seconds_elapsed(row.get("period"), pctimestring)
        shot_made = _shot_made(row)
        points = points_made_from_family(family, shot_made)
        descriptor = row.get("homedescription") or row.get("visitordescription")
        descriptor_core, _ = normalize_descriptor(descriptor)
        subfamily_raw = _subfamily(row, family, descriptor_core)
        subfamily_norm = canon_str(subfamily_raw)
        subfamily = subfamily_norm or subfamily_raw
        team_id_val = _team_id_for_row(row)
        team_tricode = _team_tricode_for_row(row)
        eventmsgtype_val = int_or_zero(row.get("eventmsgtype"))
        eventmsgactiontype_val = int_or_zero(row.get("eventmsgactiontype"))
        steal_id = int_or_zero(row.get("steal_person_id"))
        block_id = int_or_zero(row.get("block_person_id"))
        player1_team_id = int_or_zero(row.get("player1_team_id"))
        player2_team_id = int_or_zero(row.get("player2_team_id"))
        player3_team_id = int_or_zero(row.get("player3_team_id"))
        if not player1_team_id and team_id_val is not None:
            player1_team_id = int_or_zero(team_id_val)
        ft_n_val, ft_m_val = _ft_trip_from_text(row)
        game_ts = pd.to_datetime(row.get("game_date"), utc=True, errors="coerce")
        season_val = 0
        if game_ts is not None and not pd.isna(game_ts):
            ts = game_ts
            if ts.tzinfo is not None:
                ts = ts.tz_convert(None)
            season_val = get_season(ts.to_pydatetime())
        qualifiers_list: List[str] = []
        sig_key = (
            canon_str(family),
            canon_str(subfamily),
            canon_str(descriptor_core),
            tuple(sorted(canon_str(q) for q in qualifiers_list)),
        )
        overrides = mapping.get(sig_key)
        eventnum_val = int_or_zero(row.get("eventnum"))

        row_dict: Dict[str, object] = {
            "game_id": row.get("game_id"),
            "period": row.get("period"),
            "pctimestring": pctimestring,
            "seconds_elapsed": secs,
            "event_length": None,
            "action_number": eventnum_val,
            "order_number": eventnum_val,
            "eventnum": eventnum_val,
            "time_actual": row.get("game_clock"),
            "team_id": team_id_val,
            "team_tricode": team_tricode,
            "event_team": team_tricode or "",
            "player1_id": int_or_zero(row.get("player1_id")),
            "player1_name": row.get("player1_name"),
            "player1_team_id": player1_team_id,
            "player2_id": int_or_zero(row.get("player2_id")),
            "player2_name": row.get("player2_name"),
            "player2_team_id": player2_team_id,
            "player3_id": int_or_zero(row.get("player3_id")),
            "player3_name": row.get("player3_name"),
            "player3_team_id": player3_team_id,
            "home_team_id": home_id,
            "home_team_abbrev": home_tri,
            "away_team_id": away_id,
            "away_team_abbrev": away_tri,
            "homedescription": row.get("homedescription") or "",
            "visitordescription": row.get("visitordescription") or "",
            "game_date": game_ts.strftime("%Y-%m-%d") if pd.notna(game_ts) else None,
            "season": season_val,
            "family": family,
            "subfamily": subfamily,
            "eventmsgtype": eventmsgtype_val,
            "eventmsgactiontype": eventmsgactiontype_val,
            "event_type_de": EVENT_TYPE_DE.get(eventmsgtype_val, ""),
            "is_three": 1 if family == "3pt" else 0,
            "shot_made": shot_made,
            "points_made": points,
            "shot_distance": row.get("shot_distance"),
            "x": row.get("loc_x"),
            "y": row.get("loc_y"),
            "side": None,
            "area": None,
            "area_detail": None,
            "assist_id": int_or_zero(row.get("assist_person_id")),
            "block_id": block_id,
            "steal_id": steal_id,
            "style_flags": [],
            "qualifiers": qualifiers_list,
            "is_o_rebound": 1 if family == "rebound" and "OFF" in str(row.get("homedescription", "")).upper() else 0,
            "is_d_rebound": 1 if family == "rebound" and "DEF" in str(row.get("homedescription", "")).upper() else 0,
            "team_rebound": 1
            if (family == "rebound" and int_or_zero(row.get("player1_id")) == 0)
            else 0,
            "linked_shot_action_number": None,
            "possession_after": None,
            "score_home": row.get("score_home"),
            "score_away": row.get("score_away"),
            "scoremargin": scoremargin_str(row.get("score_home"), row.get("score_away")),
            "is_turnover": 1 if eventmsgtype_val == 5 else 0,
            "is_steal": 1 if eventmsgtype_val == 5 and steal_id else 0,
            "is_block": 1 if eventmsgtype_val in (1, 2) and shot_made == 0 and block_id else 0,
            "ft_n": ft_n_val,
            "ft_m": ft_m_val,
        }

        if overrides:
            if family not in {"2pt", "3pt"} and "eventmsgtype" in overrides:
                row_dict["eventmsgtype"] = int(overrides["eventmsgtype"])
            if "eventmsgactiontype" in overrides:
                row_dict["eventmsgactiontype"] = int(overrides["eventmsgactiontype"])
            if overrides.get("subfamily"):
                row_dict["subfamily"] = str(overrides["subfamily"])
            row_dict["event_type_de"] = EVENT_TYPE_DE.get(
                int_or_zero(row_dict.get("eventmsgtype")), ""
            )

        eventmsgtype_final = int_or_zero(row_dict.get("eventmsgtype"))
        row_dict["event_type_de"] = EVENT_TYPE_DE.get(eventmsgtype_final, "")
        row_dict["is_turnover"] = 1 if eventmsgtype_final == 5 else 0
        row_dict["is_steal"] = (
            1 if eventmsgtype_final == 5 and int_or_zero(row_dict.get("steal_id")) else 0
        )
        row_dict["is_block"] = (
            1
            if eventmsgtype_final in (1, 2)
            and shot_made == 0
            and int_or_zero(row_dict.get("block_id"))
            else 0
        )

        _fill_team_fields(row_dict)

        if row_dict["family"] in {"2pt", "3pt"} and (
            row_dict.get("x") is None or row_dict.get("y") is None
        ):
            synth_x, synth_y = _synth_xy(
                row_dict.get("area"), row_dict.get("area_detail"), row_dict.get("side")
            )
            if synth_x is None or synth_y is None:
                fallback = {"2pt": (50.0, 50.0), "3pt": (50.0, 82.0)}.get(
                    row_dict["family"]
                )
                if fallback:
                    synth_x, synth_y = fallback
            if synth_x is not None and synth_y is not None:
                row_dict["x"], row_dict["y"] = synth_x, synth_y
                flags = set(row_dict.get("style_flags") or [])
                flags.add("xy_synth")
                row_dict["style_flags"] = sorted(flags)

        rows.append(row_dict)

    df = pd.DataFrame(rows, columns=CANONICAL_COLUMNS)
    df = finalize_dataframe(
        df,
        sort_keys=["period", "seconds_elapsed", "order_number"],
    )
    return df
