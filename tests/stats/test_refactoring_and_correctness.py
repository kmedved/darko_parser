import pandas as pd
import numpy as np
import pytest
from nba_scraper.stats import PbP
from nba_scraper.stats.box_glossary import (
    accumulate_player_counts,
    annotate_events,
    build_player_box,
)


def test_string_ids_are_normalized(v2_pbp_df):
    """Ensure PbP and downstream box glossary coerce string IDs to ints."""

    pbp_df = v2_pbp_df.copy()
    id_cols = [
        "game_id",
        "team_id",
        "home_team_id",
        "away_team_id",
        *(f"home_player_{i}_id" for i in range(1, 6)),
        *(f"away_player_{i}_id" for i in range(1, 6)),
        "player1_id",
        "player2_id",
        "player3_id",
        "player1_team_id",
        "player2_team_id",
        "player3_team_id",
    ]
    for col in id_cols:
        if col in pbp_df.columns:
            pbp_df[col] = pbp_df[col].astype(str)
    pbp_df["season"] = 2008

    box = PbP(pbp_df).player_box_glossary()

    assert pd.api.types.is_integer_dtype(box["game_id"])
    assert pd.api.types.is_integer_dtype(box["team_id"])
    assert pd.api.types.is_integer_dtype(box["player_id"])


def test_decoupling_from_pbg_stats(pbp_v2_game):
    """
    Validates Change 1:
    Ensures box score stats are correct after removing the pbg_stats override.
    Also checks that players with minutes but no stats are now included.
    """
    box = pbp_v2_game.player_box_glossary()
    
    # Check a known player's stats (Carmelo Anthony)
    melo = box[box["player_id"] == 2546]
    assert melo["FGM"].iloc[0] == 7
    assert melo["FGA"].iloc[0] == 15
    assert melo["ThreePM"].iloc[0] == 0
    assert melo["ThreePA"].iloc[0] == 1
    assert melo["FTM"].iloc[0] == 10
    assert melo["FTA"].iloc[0] == 11
    assert melo["PTS"].iloc[0] == 24

    # This test file doesn't have a zero-stat player, but we confirm
    # that the number of players matches the number with non-zero minutes.
    player_rows = box[box["Player_Team"] != "TOTAL"]

    players_in_box = len(player_rows)
    players_with_minutes = len(player_rows[player_rows["Minutes"] > 0])
    assert players_in_box == players_with_minutes


def test_team_totals_appended_by_default(pbp_v2_game):
    box = pbp_v2_game.player_box_glossary()

    totals = box[box["Player_Team"] == "TOTAL"]
    player_rows = box[box["Player_Team"] != "TOTAL"]

    assert len(totals) == 2

    # Team identifiers should mirror the team_id
    assert (totals["player_id"] == totals["team_id"]).all()
    assert (totals["NbaDotComID"] == totals["team_id"]).all()
    assert (totals["PlayerID"] == totals["team_id"]).all()
    assert (totals["PlayerSeasonID"] == totals["team_id"]).all()

    for team_id, total_row in totals.groupby("team_id"):
        team_players = player_rows[player_rows["team_id"] == team_id]

        assert pytest.approx(team_players["Minutes"].sum()) == total_row["Minutes"].iloc[0]
        assert pytest.approx(team_players["PTS"].sum()) == total_row["PTS"].iloc[0]
        assert pytest.approx(team_players["TOV"].sum()) == total_row["TOV"].iloc[0]


def test_vectorized_helpers_match_row_wise():
    """
    Validates Change 3:
    Ensures the new vectorized helpers for and-one and shot-zone produce
    the same output as the old row-wise logic.
    """
    df = pd.DataFrame({
        "qualifiers": ["['and1']", "['other']", "['and-one']"],
        "shot_distance": [2, 8, 25],
        "area": ["Restricted Area", "In The Paint (Non-RA)", "Mid-Range"],
        "is_fg_attempt": [True, True, False] # Last one is not a shot
    })

    # Old logic for comparison
    from nba_scraper.stats.box_glossary import classify_shot_zone
    df["shot_zone_old"] = df.apply(lambda r: classify_shot_zone(r.get("shot_distance"), r.get("area")), axis=1)
    df["is_and_one_old"] = df["qualifiers"].astype(str).str.lower().str.contains(r"and[ -]?one|and1")

    # New vectorized logic
    annotated = annotate_events(df.copy())

    pd.testing.assert_series_equal(annotated["shot_zone"], df["shot_zone_old"], check_names=False)
    pd.testing.assert_series_equal(annotated["is_and_one"], df["is_and_one_old"], check_names=False)


def test_assist_attribution_fix():
    """
    Validates Change 4:
    Ensures assists are not credited on missed shots.
    """
    pbp_events = pd.DataFrame({
        "game_id": ["g1", "g1"],
        "player1_id": [101, 101], # Shooter
        "player1_team_id": [1, 1],
        "player2_id": [202, 202], # Helper on make, defender on miss
        "team_id": [1, 1],
        "is_fg_attempt": [True, True],
        "is_fg_make": [True, False], # One make, one miss
        "assist_id": [202, np.nan], # Assist on make, no assist on miss
    })
    
    counts_df = accumulate_player_counts(pbp_events)
    
    # Player 202 should have exactly 1 assist from the made shot
    assister_stats = counts_df[counts_df["player_id"] == 202]
    assert not assister_stats.empty
    assert assister_stats["AST"].iloc[0] == 1


def test_metadata_passthrough(pbp_v2_game):
    """
    Validates Change 5:
    Ensures that pre-existing metadata columns like 'Starts' are preserved
    and not overwritten with defaults.
    """
    # Create player_meta where a player is manually marked as a starter
    player_meta = pd.DataFrame({
        "player_id": [200784],
        "season": [2008],
        "Starts": [1] # Override default
    })
    
    box = pbp_v2_game.player_box_glossary(player_meta=player_meta)
    
    diawara_row = box[box["player_id"] == 200784]
    
    # The 'Starts' value from player_meta should be preserved
    assert not diawara_row.empty
    assert diawara_row["Starts"].iloc[0] == 1


def test_player_box_glossary_cdn_game(cdn_0021900151_df):
    """
    Smoke + invariants test on a CDN-era game to ensure the new pipeline
    works for modern data as well.
    """
    pbp = PbP(cdn_0021900151_df)
    box = pbp.player_box_glossary()

    player_rows = box[box["Player_Team"] != "TOTAL"]
    totals = box[box["Player_Team"] == "TOTAL"]

    assert len(totals) == 2

    # Minutes sanity: each team should be ~game_minutes * 5
    total_minutes = cdn_0021900151_df["seconds_elapsed"].max() / 60.0
    team_minutes = player_rows.groupby("team_id")["Minutes"].sum()
    for minutes in team_minutes:
        assert abs(minutes - total_minutes * 5.0) < 1.0

    # On-court scoring invariants: sum of per-player on/off points should
    # match the scoreboard totals derived from canonical points_made.
    points = pd.to_numeric(cdn_0021900151_df["points_made"], errors="coerce").fillna(0)
    points_df = pd.DataFrame({
        "team_id": cdn_0021900151_df["team_id"],
        "points": points,
    })
    points_df = points_df[(points_df["team_id"].notna()) & (points_df["team_id"] != 0)]
    team_points = points_df.groupby("team_id")["points"].sum()
    team_points_map = {int(team): float(val) for team, val in team_points.items()}

    assert team_points_map, "Fixture should include at least one scoring team"

    for team_id, pts_for in team_points_map.items():
        on_for = player_rows.loc[player_rows["team_id"] == team_id, "OnCourt_Team_Points"].sum()
        opp_pts = sum(p for t, p in team_points_map.items() if t != team_id)
        on_against = player_rows.loc[player_rows["team_id"] == team_id, "OnCourt_Opp_Points"].sum()

        assert abs(on_for - pts_for * 5.0) < 1e-6
        assert abs(on_against - opp_pts * 5.0) < 1e-6
