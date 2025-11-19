import pandas as pd

from nba_scraper.stats import PbP


def test_class_build(pbp_v2_game):
    """Ensure the PbP wrapper exposes key metadata from the fixture."""

    pbp = pbp_v2_game

    assert isinstance(pbp, PbP)
    assert isinstance(pbp.df, pd.DataFrame)
    assert pbp.home_team == "DEN"
    assert pbp.away_team == "LAC"
    assert pbp.home_team_id == 1610612743
    assert pbp.away_team_id == 1610612746
    assert pbp.game_date.strftime("%Y-%m-%d") == "2007-11-30"
    assert pbp.season == 2008


def test_point_calc_player(pbp_v2_game):
    """Regression coverage for the point/shot aggregation helper."""

    stats_df = pbp_v2_game._point_calc_player()

    ai = stats_df.loc[stats_df["player_id"] == 1894].iloc[0]
    carmelo = stats_df.loc[stats_df["player_id"] == 947].iloc[0]

    assert ai["fgm"] == 10
    assert ai["fga"] == 20
    assert ai["tpm"] == 1
    assert ai["tpa"] == 2
    assert ai["ftm"] == 5
    assert ai["fta"] == 6

    assert carmelo["fgm"] == 11
    assert carmelo["fga"] == 26
    assert carmelo["tpm"] == 0
    assert carmelo["tpa"] == 2
    assert carmelo["ftm"] == 4
    assert carmelo["fta"] == 6


def test_player_misc_counts(pbp_v2_game):
    """Check a representative slice of per-player non-scoring stats."""

    blocks = pbp_v2_game._block_calc_player()
    assists = pbp_v2_game._assist_calc_player()
    rebounds = pbp_v2_game._rebound_calc_player()
    steals = pbp_v2_game._steal_calc_player()
    turnovers = pbp_v2_game._turnover_calc_player()
    fouls = pbp_v2_game._foul_calc_player()

    assert blocks.loc[blocks["player_id"] == 2549, "blk"].iloc[0] == 3
    assert assists.loc[assists["player_id"] == 1894, "ast"].iloc[0] == 5

    iverson_rebs = rebounds.loc[rebounds["player_id"] == 1894].iloc[0]
    assert iverson_rebs["dreb"] == 2
    assert iverson_rebs["oreb"] == 2

    assert steals.loc[steals["player_id"] == 1510, "stl"].iloc[0] == 2
    assert turnovers.loc[turnovers["player_id"] == 1894, "tov"].iloc[0] == 6
    assert fouls.loc[fouls["player_id"] == 2546, "pf"].iloc[0] == 4


def test_plus_minus_calc_player(pbp_v2_game):
    """Plus/minus should align with known historical totals."""

    plus_minus = pbp_v2_game._plus_minus_calc_player()
    stats_df = pbp_v2_game._point_calc_player()

    merged = stats_df.merge(
        plus_minus, how="left", on=["player_id", "team_id", "game_date", "game_id"]
    ).fillna({"plus_minus": 0})

    assert merged.loc[merged["player_id"] == 1894, "plus_minus"].iloc[0] == -8
    assert merged.loc[merged["player_id"] == 947, "plus_minus"].iloc[0] == 16
    assert merged.loc[merged["player_id"] == 2546, "plus_minus"].iloc[0] == 14


def test_toc_calc_player(pbp_v2_game):
    """Time-on-court helper should compute both raw seconds and clock strings."""

    toc = pbp_v2_game._toc_calc_player()

    ai = toc.loc[toc["player_id"] == 1894].iloc[0]
    carmelo = toc.loc[toc["player_id"] == 947].iloc[0]

    assert ai["toc"] == 2307
    assert ai["toc_string"] == "38:27"
    assert carmelo["toc"] == 2452
    assert carmelo["toc_string"] == "40:52"


def test_playerbygamestats(pbp_v2_game):
    """High-level regression for the legacy playerbygamestats dataframe."""

    pbg = pbp_v2_game.playerbygamestats()

    assert pbg.loc[pbg["player_id"] == 1894, "plus_minus"].iloc[0] == -8
    assert pbg.loc[pbg["player_id"] == 947, "plus_minus"].iloc[0] == 16
    assert pbg.loc[pbg["player_id"] == 2546, "pf"].iloc[0] == 4
    assert pbg.loc[pbg["player_id"] == 1510, "stl"].iloc[0] == 2
    assert pbg.loc[pbg["player_id"] == 2059, "oreb"].iloc[0] == 2
    assert pbg.loc[pbg["player_id"] == 1894, "toc"].iloc[0] == 2307
    assert pbg.loc[pbg["player_id"] == 1894, "opponent"].iloc[0] == 1610612743
    assert pbg.loc[pbg["player_id"] == 947, "opponent_abbrev"].iloc[0] == "LAC"


def test_calc_point_team(pbp_v2_game):
    """Team point calculations should match the official box totals."""

    points = pbp_v2_game._point_calc_team()
    clip = points.loc[points["team_id"] == 1610612746].iloc[0]
    nuggets = points.loc[points["team_id"] == 1610612743].iloc[0]

    assert clip["points_for"] == 107
    assert nuggets["points_for"] == 123
    assert clip["points_for"] + nuggets["points_for"] == points["points_for"].sum()


def test_teambygamestats(pbp_v2_game):
    """Legacy team aggregates should mirror the player-based totals."""

    tbg = pbp_v2_game.teambygamestats()

    summary = tbg.drop_duplicates(subset=["team_id"])
    clips = summary.loc[summary["team_id"] == 1610612746].iloc[0]
    nuggets = summary.loc[summary["team_id"] == 1610612743].iloc[0]

    assert clips["points_for"] == 107
    assert nuggets["points_for"] == 123
    assert clips["fgm"] == 37
    assert nuggets["fgm"] == 46
    assert clips["tpm"] == 7
    assert nuggets["tpm"] == 10
    assert clips["ftm"] == 26
    assert nuggets["ftm"] == 21
    assert clips["oreb"] == 6
    assert nuggets["oreb"] == 9
    assert clips["opponent"] == 1610612743
    assert nuggets["opponent_abbrev"] == "LAC"
