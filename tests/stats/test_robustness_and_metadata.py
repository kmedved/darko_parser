import pandas as pd
import numpy as np
import pytest
from nba_scraper.stats import PbP
from nba_scraper.stats.box_glossary import annotate_events


def test_pbp_enforces_single_game():
    """Validates Change 1: PbP class raises ValueError for multi-game DFs."""
    multi_game_df = pd.DataFrame({
        "game_id": ["001", "002"],
        "home_team_abbrev": ["A", "C"],
        "away_team_abbrev": ["B", "D"],
        "home_team_id": [1, 3],
        "away_team_id": [2, 4],
        "season": [2024, 2024],
        "game_date": ["2024-01-01", "2024-01-02"],
        "scoremargin": ["0", "0"],
    })
    with pytest.raises(ValueError, match="expects a single-game DataFrame"):
        PbP(multi_game_df)


def test_robust_team_id_normalization():
    """Validates Change 2: annotate_events correctly fills missing/zero team_id."""
    df = pd.DataFrame({
        "team_id": [1, 0, np.nan],  # One correct, one zero, one NaN
        "event_team": ["HOME", "AWAY", "HOME"],
        "home_team_abbrev": ["HOME", "HOME", "HOME"],
        "away_team_abbrev": ["AWAY", "AWAY", "AWAY"],
        "home_team_id": [1, 1, 1],
        "away_team_id": [2, 2, 2],
    })

    annotated = annotate_events(df)

    assert annotated["team_id"].tolist() == [1, 2, 1]


def test_possession_after_guard_falls_back_to_legacy_flags():
    home_id = 10
    away_id = 20

    df = pd.DataFrame(
        [
            {
                "game_id": 12345678,
                "period": 1,
                "pctimestring": "12:00",
                "seconds_elapsed": 0,
                "event_type_de": "shot",
                "family": "shot",
                "event_team": "HOM",
                "team_id": home_id,
                "home_team_abbrev": "HOM",
                "away_team_abbrev": "AWY",
                "home_team_id": home_id,
                "away_team_id": away_id,
                "game_date": "2024-01-01",
                "season": 2024,
                "scoremargin": 0,
                "possession_after": 0,
                "is_d_rebound": 0,
                "is_o_rebound": 0,
            },
            {
                "game_id": 12345678,
                "period": 1,
                "pctimestring": "11:59",
                "seconds_elapsed": 1,
                "event_type_de": "turnover",
                "family": "turnover",
                "event_team": "AWY",
                "team_id": away_id,
                "home_team_abbrev": "HOM",
                "away_team_abbrev": "AWY",
                "home_team_id": home_id,
                "away_team_id": away_id,
                "game_date": "2024-01-01",
                "season": 2024,
                "scoremargin": 0,
                "possession_after": None,
                "is_d_rebound": 0,
                "is_o_rebound": 0,
            },
        ]
    )

    pbp = PbP(df)

    assert pbp.df["home_possession"].sum() == 1
    assert pbp.df["away_possession"].sum() == 1


def test_possession_after_zero_nan_still_flags_possessions():
    home_id = 30
    away_id = 40

    df = pd.DataFrame(
        [
            {
                "game_id": 9876543,
                "period": 1,
                "pctimestring": "12:00",
                "seconds_elapsed": 0,
                "event_type_de": "shot",
                "family": "shot",
                "event_team": "HOM",
                "team_id": home_id,
                "home_team_abbrev": "HOM",
                "away_team_abbrev": "AWY",
                "home_team_id": home_id,
                "away_team_id": away_id,
                "game_date": "2024-01-01",
                "season": 2024,
                "scoremargin": 0,
                "possession_after": 0,
                "is_d_rebound": 0,
                "is_o_rebound": 0,
            },
            {
                "game_id": 9876543,
                "period": 1,
                "pctimestring": "11:59",
                "seconds_elapsed": 1,
                "event_type_de": "turnover",
                "family": "turnover",
                "event_team": "AWY",
                "team_id": away_id,
                "home_team_abbrev": "HOM",
                "away_team_abbrev": "AWY",
                "home_team_id": home_id,
                "away_team_id": away_id,
                "game_date": "2024-01-01",
                "season": 2024,
                "scoremargin": 0,
                "possession_after": np.nan,
                "is_d_rebound": 0,
                "is_o_rebound": 0,
            },
        ]
    )

    pbp = PbP(df)

    assert pbp.df["home_possession"].sum() == 1
    assert pbp.df["away_possession"].sum() == 1


def test_possession_after_only_zero_nan_defaults_to_heuristics():
    home_id = 50
    away_id = 60

    df = pd.DataFrame(
        [
            {
                "game_id": 24681012,
                "period": 1,
                "pctimestring": "12:00",
                "seconds_elapsed": 0,
                "event_type_de": "shot",
                "family": "shot",
                "event_team": "HOM",
                "team_id": home_id,
                "home_team_abbrev": "HOM",
                "away_team_abbrev": "AWY",
                "home_team_id": home_id,
                "away_team_id": away_id,
                "game_date": "2024-01-01",
                "season": 2024,
                "scoremargin": 0,
                "possession_after": 0,
                "is_d_rebound": 0,
                "is_o_rebound": 0,
            },
            {
                "game_id": 24681012,
                "period": 1,
                "pctimestring": "11:58",
                "seconds_elapsed": 2,
                "event_type_de": "turnover",
                "family": "turnover",
                "event_team": "AWY",
                "team_id": away_id,
                "home_team_abbrev": "HOM",
                "away_team_abbrev": "AWY",
                "home_team_id": home_id,
                "away_team_id": away_id,
                "game_date": "2024-01-01",
                "season": 2024,
                "scoremargin": 0,
                "possession_after": np.nan,
                "is_d_rebound": 0,
                "is_o_rebound": 0,
            },
            {
                "game_id": 24681012,
                "period": 1,
                "pctimestring": "11:56",
                "seconds_elapsed": 4,
                "event_type_de": "turnover",
                "family": "turnover",
                "event_team": "HOM",
                "team_id": home_id,
                "home_team_abbrev": "HOM",
                "away_team_abbrev": "AWY",
                "home_team_id": home_id,
                "away_team_id": away_id,
                "game_date": "2024-01-01",
                "season": 2024,
                "scoremargin": 0,
                "possession_after": 0,
                "is_d_rebound": 0,
                "is_o_rebound": 0,
            },
        ]
    )

    pbp = PbP(df)

    assert pbp.df["home_possession"].tolist() == [1, 0, 1]
    assert pbp.df["away_possession"].tolist() == [0, 1, 0]


def test_compute_starters(pbp_v2_game):
    """Validates Change 3: Starters are correctly identified."""
    box = pbp_v2_game.player_box_glossary()

    # From the 20700233.csv file, we can identify the starters
    denver_starters = {2546, 2030, 948, 947, 1853}
    lac_starters = {1739, 1501, 2549, 1894, 1510}
    all_starters = denver_starters.union(lac_starters)

    starter_rows = box[box["Starts"] == 1]
    non_starter_rows = box[box["Starts"] == 0]

    assert len(starter_rows) == 10
    assert set(starter_rows["player_id"]) == all_starters
    # Ensure no non-starters were incorrectly flagged
    assert not any(pid in all_starters for pid in non_starter_rows["player_id"])


def test_player_meta_deduplication(pbp_v2_game):
    """Validates Change 4: Multi-season player_meta is correctly filtered."""
    # Player 947 has rows for two seasons
    multi_season_meta = pd.DataFrame({
        "player_id": [947, 947, 1894],
        "season": [2008, 2009, 2008],
        "FullName": ["Allen Iverson (08)", "Allen Iverson (09)", "Cuttino Mobley"],
        "Age": [32, 33, 32]
    })

    # The game's season is 2008
    box = pbp_v2_game.player_box_glossary(player_meta=multi_season_meta)

    ai_row = box[box["player_id"] == 947]

    # Check that we only have one row for the player
    assert len(ai_row) == 1
    # Check that the correct season's data was used
    assert ai_row["FullName"].iloc[0] == "Allen Iverson (08)"
    assert ai_row["Age"].iloc[0] == 32
