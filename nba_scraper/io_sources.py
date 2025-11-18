"""Source routing helpers for the unified parsing pipeline."""
from __future__ import annotations

import json
import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd

from . import cdn_client, cdn_parser, lineup_builder, v2_parser
from .coords_backfill import backfill_coords_with_shotchart
from .boxscore_validation import log_team_boxscore_mismatches

_BACKFILL_COORDS = os.getenv("NBA_SCRAPER_BACKFILL_COORDS", "0") == "1"
_VALIDATE_BOX = os.getenv("NBA_SCRAPER_VALIDATE_BOX", "0") == "1"


class SourceKind(str, Enum):
    CDN_REMOTE = "cdn_remote"
    CDN_LOCAL = "cdn_local"
    V2_LOCAL = "v2_local"
    V2_DICT = "v2_dict"


def load_json(path_or_dict: Union[str, Path, Dict[str, Any]], kind: SourceKind) -> Dict[str, Any]:
    if kind == SourceKind.V2_DICT and isinstance(path_or_dict, dict):
        return path_or_dict
    if kind in {SourceKind.CDN_LOCAL, SourceKind.V2_LOCAL}:
        path = Path(path_or_dict)
        with path.open("r", encoding="utf-8") as fp:
            return json.load(fp)
    raise ValueError(f"Unsupported load_json kind: {kind}")


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value in (None, "", []):
            continue
        return value
    return None


def _shotchart_payload_to_df(payload: Dict[str, Any]) -> pd.DataFrame:
    shots: list[dict[str, Any]] = []

    def _collect_shots(team_blob: Dict[str, Any]) -> None:
        for shot in (team_blob or {}).get("shots", []) or []:
            game_id = _coalesce(
                shot.get("gameId"),
                shot.get("game_id"),
                payload.get("gameId"),
                payload.get("game_id"),
            )
            eventnum = _coalesce(
                shot.get("gameEventId"),
                shot.get("eventnum"),
                shot.get("eventNum"),
                shot.get("eventId"),
                shot.get("actionNumber"),
                shot.get("orderNumber"),
            )
            x_val = _coalesce(shot.get("x"), shot.get("shotX"), shot.get("shot_x"))
            y_val = _coalesce(shot.get("y"), shot.get("shotY"), shot.get("shot_y"))
            dist = _coalesce(
                shot.get("shotDistance"),
                shot.get("shot_distance"),
                shot.get("distance"),
            )
            shots.append(
                {
                    "game_id": game_id,
                    "eventnum": eventnum,
                    "x": x_val,
                    "y": y_val,
                    "shot_distance": dist,
                }
            )

    if not isinstance(payload, dict):
        return pd.DataFrame(columns=["game_id", "eventnum", "x", "y", "shot_distance"])

    if "shots" in payload and isinstance(payload.get("shots"), list):
        _collect_shots({"shots": payload["shots"]})

    game_section = (payload or {}).get("game", {})
    if "teams" in payload:
        for team_blob in (payload.get("teams") or []):
            if isinstance(team_blob, dict):
                _collect_shots(team_blob)
    for side in ("homeTeam", "awayTeam"):
        team_blob = game_section.get(side)
        if isinstance(team_blob, dict):
            _collect_shots(team_blob)

    if not shots:
        return pd.DataFrame(columns=["game_id", "eventnum", "x", "y", "shot_distance"])

    return pd.DataFrame(shots, columns=["game_id", "eventnum", "x", "y", "shot_distance"])


def _fetch_shotchart_df(game_id: str) -> Optional[pd.DataFrame]:
    try:
        payload = cdn_client.fetch_shotchart(game_id)
    except Exception:
        return None
    df = _shotchart_payload_to_df(payload)
    return df if not df.empty else None


def _load_local_shotchart(game_ref: Tuple[Union[str, Path], Optional[Union[str, Path]]]) -> Optional[pd.DataFrame]:
    pbp_ref, _ = game_ref
    if isinstance(pbp_ref, dict):
        payload = pbp_ref.get("shotchart") if isinstance(pbp_ref, dict) else None
        if isinstance(payload, dict):
            df = _shotchart_payload_to_df(payload)
            return df if not df.empty else None
        return None
    pbp_path = Path(pbp_ref)
    candidates = [
        pbp_path.with_name(pbp_path.name.replace("playbyplay", "shotchart")),
        pbp_path.with_suffix(".shotchart.json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            try:
                payload = load_json(candidate, SourceKind.CDN_LOCAL)
            except Exception:
                continue
            df = _shotchart_payload_to_df(payload)
            if not df.empty:
                return df
    return None


def parse_any(
    game_ref: Union[str, Tuple[Union[str, Path], Optional[Union[str, Path]]], Dict[str, Any]],
    kind: SourceKind,
    mapping_yaml_path: Optional[str] = None,
) -> pd.DataFrame:
    if kind == SourceKind.CDN_REMOTE:
        if not isinstance(game_ref, str):
            raise TypeError("CDN_REMOTE requires a game id string")
        pbp_json = cdn_client.fetch_pbp(game_ref)
        box_json = None
        try:
            box_json = cdn_client.fetch_box(game_ref)
        except Exception:
            box_json = None
        meta_json_for_parser = box_json or pbp_json
        df = cdn_parser.parse_actions_to_rows(
            pbp_json,
            meta_json_for_parser or {},
            mapping_yaml_path,
        )
        if _BACKFILL_COORDS:
            shotchart_df = _fetch_shotchart_df(game_ref)
            if shotchart_df is not None:
                df = backfill_coords_with_shotchart(df, shotchart_df)
        df = lineup_builder.attach_lineups(
            df, box_json=box_json, pbp_json=pbp_json
        )
        if _VALIDATE_BOX and box_json is not None:
            # Log any core-stat mismatches between PbP-derived totals and boxscore.
            log_team_boxscore_mismatches(df, box_json)
        return df

    if kind == SourceKind.CDN_LOCAL:
        if not isinstance(game_ref, (tuple, list)) or len(game_ref) != 2:
            raise TypeError("CDN_LOCAL requires (pbp_path, box_path)")
        pbp_ref, box_ref = game_ref
        pbp_json = (
            pbp_ref
            if isinstance(pbp_ref, dict)
            else load_json(pbp_ref, SourceKind.CDN_LOCAL)
        )
        if box_ref is None:
            raise TypeError("CDN_LOCAL requires box score path")
        box_json = (
            box_ref
            if isinstance(box_ref, dict)
            else load_json(box_ref, SourceKind.CDN_LOCAL)
        )
        df = cdn_parser.parse_actions_to_rows(pbp_json, box_json, mapping_yaml_path)
        if _BACKFILL_COORDS:
            shotchart_df = _load_local_shotchart((pbp_ref, box_ref))
            if shotchart_df is not None:
                df = backfill_coords_with_shotchart(df, shotchart_df)
        df = lineup_builder.attach_lineups(
            df, box_json=box_json, pbp_json=pbp_json
        )
        if _VALIDATE_BOX and box_json is not None:
            # Log any core-stat mismatches between PbP-derived totals and boxscore.
            log_team_boxscore_mismatches(df, box_json)
        return df

    if kind == SourceKind.V2_LOCAL:
        if not isinstance(game_ref, (str, Path)):
            raise TypeError("V2_LOCAL requires a path")
        v2_json = load_json(game_ref, SourceKind.V2_LOCAL)
        df = v2_parser.parse_v2_to_rows(v2_json, mapping_yaml_path)
        return lineup_builder.attach_lineups(df)

    if kind == SourceKind.V2_DICT:
        if not isinstance(game_ref, dict):
            raise TypeError("V2_DICT requires a dictionary")
        df = v2_parser.parse_v2_to_rows(game_ref, mapping_yaml_path)
        return lineup_builder.attach_lineups(df)

    raise ValueError(f"Unsupported source kind: {kind}")
