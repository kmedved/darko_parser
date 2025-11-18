from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("pandas")

from nba_scraper import boxscore_validation, cdn_parser  # noqa: E402


FIXTURES = Path(__file__).parent / "test_files"


def _load_json(name: str):
    with (FIXTURES / name).open(encoding="utf-8") as fh:
        return json.load(fh)


def test_boxscore_validation_matches_fixture():
    """
    Sanity check: for the fixture game, PbP-derived team totals should match
    the official boxscore for the shooting and counting stats we expose.

    This acts as a regression harness for future parser changes that might
    accidentally drop or misclassify events.
    """
    pbp = _load_json("cdn_playbyplay_0022400001.json")
    box = _load_json("cdn_boxscore_0022400001.json")

    df = cdn_parser.parse_actions_to_rows(pbp, box)
    fields = boxscore_validation.EXTENDED_TEAM_STAT_FIELDS
    mismatches = boxscore_validation.compare_pbp_to_box(df, box, fields=fields, atol=0)

    assert mismatches == []


def test_boxscore_validation_handles_empty_inputs():
    """
    Defensive behavior on empty dataframes / malformed boxscore payloads.
    """
    import pandas as pd

    empty_df = pd.DataFrame(
        columns=["game_id", "team_id", "family", "shot_made", "points_made"]
    )
    mismatches = boxscore_validation.compare_pbp_to_box(empty_df, {}, atol=0)
    assert mismatches == []


def test_boxscore_validation_extended_fields():
    import pandas as pd

    df = pd.DataFrame(
        [
            {
                "team_id": 1,
                "family": "2pt",
                "shot_made": 1,
                "points_made": 2,
                "assist_id": 11,
                "is_turnover": 0,
                "is_steal": 0,
                "is_block": 0,
                "is_o_rebound": 0,
                "is_d_rebound": 0,
                "team_rebound": 0,
                "player2_team_id": 0,
                "player3_team_id": 0,
            },
            {
                "team_id": 1,
                "family": "2pt",
                "shot_made": 0,
                "points_made": 0,
                "assist_id": 0,
                "is_turnover": 0,
                "is_steal": 0,
                "is_block": 1,
                "is_o_rebound": 0,
                "is_d_rebound": 0,
                "team_rebound": 0,
                "player2_team_id": 0,
                "player3_team_id": 2,
            },
            {
                "team_id": 2,
                "family": "3pt",
                "shot_made": 0,
                "points_made": 0,
                "assist_id": 0,
                "is_turnover": 0,
                "is_steal": 0,
                "is_block": 1,
                "is_o_rebound": 0,
                "is_d_rebound": 0,
                "team_rebound": 0,
                "player2_team_id": 0,
                "player3_team_id": 1,
            },
            {
                "team_id": 2,
                "family": "2pt",
                "shot_made": 1,
                "points_made": 2,
                "assist_id": 7,
                "is_turnover": 0,
                "is_steal": 0,
                "is_block": 0,
                "is_o_rebound": 0,
                "is_d_rebound": 0,
                "team_rebound": 0,
                "player2_team_id": 0,
                "player3_team_id": 0,
            },
            {
                "team_id": 1,
                "family": "rebound",
                "shot_made": 0,
                "points_made": 0,
                "assist_id": 0,
                "is_turnover": 0,
                "is_steal": 0,
                "is_block": 0,
                "is_o_rebound": 1,
                "is_d_rebound": 0,
                "team_rebound": 0,
                "player2_team_id": 0,
                "player3_team_id": 0,
            },
            {
                "team_id": 1,
                "family": "rebound",
                "shot_made": 0,
                "points_made": 0,
                "assist_id": 0,
                "is_turnover": 0,
                "is_steal": 0,
                "is_block": 0,
                "is_o_rebound": 0,
                "is_d_rebound": 0,
                "team_rebound": 1,
                "player2_team_id": 0,
                "player3_team_id": 0,
            },
            {
                "team_id": 2,
                "family": "rebound",
                "shot_made": 0,
                "points_made": 0,
                "assist_id": 0,
                "is_turnover": 0,
                "is_steal": 0,
                "is_block": 0,
                "is_o_rebound": 0,
                "is_d_rebound": 1,
                "team_rebound": 0,
                "player2_team_id": 0,
                "player3_team_id": 0,
            },
            {
                "team_id": 1,
                "family": "turnover",
                "shot_made": 0,
                "points_made": 0,
                "assist_id": 0,
                "is_turnover": 1,
                "is_steal": 1,
                "is_block": 0,
                "is_o_rebound": 0,
                "is_d_rebound": 0,
                "team_rebound": 0,
                "player2_team_id": 2,
                "player3_team_id": 0,
            },
            {
                "team_id": 2,
                "family": "turnover",
                "shot_made": 0,
                "points_made": 0,
                "assist_id": 0,
                "is_turnover": 1,
                "is_steal": 1,
                "is_block": 0,
                "is_o_rebound": 0,
                "is_d_rebound": 0,
                "team_rebound": 0,
                "player2_team_id": 1,
                "player3_team_id": 0,
            },
        ]
    )

    box = {
        "game": {
            "homeTeam": {
                "teamId": 1,
                "score": 2,
                "statistics": {
                    "points": 2,
                    "fieldGoalsMade": 1,
                    "fieldGoalsAttempted": 2,
                    "threePointersMade": 0,
                    "threePointersAttempted": 0,
                    "freeThrowsMade": 0,
                    "freeThrowsAttempted": 0,
                    "reboundsTotal": 2,
                    "assists": 1,
                    "steals": 1,
                    "blocks": 1,
                    "turnovers": 1,
                },
            },
            "awayTeam": {
                "teamId": 2,
                "score": 2,
                "statistics": {
                    "points": 2,
                    "fieldGoalsMade": 1,
                    "fieldGoalsAttempted": 2,
                    "threePointersMade": 0,
                    "threePointersAttempted": 1,
                    "freeThrowsMade": 0,
                    "freeThrowsAttempted": 0,
                    "reboundsTotal": 1,
                    "assists": 1,
                    "steals": 1,
                    "blocks": 1,
                    "turnovers": 1,
                },
            },
        }
    }

    mismatches = boxscore_validation.compare_pbp_to_box(
        df, box, fields=boxscore_validation.EXTENDED_TEAM_STAT_FIELDS, atol=0
    )

    assert mismatches == []
