import pandas as pd
from nba_parser.teamtotals import TeamTotals


def test_team_ft_metrics_match_definitions():
    # Synthetic one-game sample:
    # Team: 5 FTM on 10 FTA, 20 FGA  -> FT% = 0.5, FTM/FGA = 0.25
    # Opp : 6 FTM on  8 FTA, 16 FGA  -> FT% = 0.75, FTM/FGA = 0.375
    team_row = {
        "team_id": 1,
        "team_abbrev": "AAA",
        "fgm": 10,
        "tpm": 0,
        "fga": 20,
        "points_for": 20,
        "points_against": 16,
        "plus_minus": 4,
        "tpa": 0,
        "fta": 10,
        "tov": 0,
        "dreb": 0,
        "oreb": 0,
        "ftm": 5,
        "ast": 0,
        "blk": 0,
        "possessions": 80,
        "game_id": "TEST",
        "season": 2024,
    }
    opponent_row = {
        "team_id": 2,
        "team_abbrev": "BBB",
        "fgm": 8,
        "tpm": 0,
        "fga": 16,
        "points_for": 16,
        "points_against": 20,
        "plus_minus": -4,
        "tpa": 0,
        "fta": 8,
        "tov": 0,
        "dreb": 0,
        "oreb": 0,
        "ftm": 6,
        "ast": 0,
        "blk": 0,
        "possessions": 80,
        "game_id": "TEST",
        "season": 2024,
    }
    tbg = pd.DataFrame([team_row, opponent_row])
    tt = TeamTotals([tbg])
    result = tt.team_advanced_stats().set_index("team_id").loc[1]

    assert result["ft_percent"] == 5 / 10
    assert result["opp_ft_percent"] == 6 / 8
    assert result["ft_per_fga"] == 5 / 20
    assert result["opp_ft_per_fga"] == 6 / 16
