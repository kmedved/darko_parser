import numpy as np
import pandas as pd

# --- CHANGED: Point to the new internal stats module ---
from nba_scraper.stats import box_glossary


def _mini_true_shooting_pbp():
    base = {
        "game_id": 1,
        "home_team_id": 1,
        "away_team_id": 2,
        "home_team_abbrev": "HOM",
        "away_team_abbrev": "AWY",
        "event_team": "HOM",
        "player1_id": 10,
        "player1_team_id": 1,
    }

    return pd.DataFrame(
        [
            {
                **base,
                "event_type_de": "shot",
                "subfamily_de": "jumper",
                "qualifiers": "and1",
                "points_made": 2,
                "is_three": False,
                "shot_made": 1,
            },
            {
                **base,
                "event_type_de": "free_throw",
                "subfamily_de": "1 of 1",
                "points_made": 1,
            },
            {
                **base,
                "event_type_de": "free_throw",
                "subfamily_de": "Technical",
                "points_made": 1,
            },
        ]
    )

def test_vectorized_shot_zone_area_only():
    """
    _vectorized_shot_zone should not crash when shot_distance is missing,
    and should fall back to area-based classification.
    """
    df = pd.DataFrame(
        {
            # No usable distance info
            "shot_distance": [np.nan, np.nan, np.nan],
            "area": [
                "Restricted Area",
                "In The Paint (Non-RA)",
                "Some Other Zone",
            ],
        }
    )

    zones = box_glossary._vectorized_shot_zone(df)

    assert zones.tolist() == ["0_3", "4_9", None]


def test_vectorized_is_and_one_handles_various_qualifier_types():
    """
    _vectorized_is_and_one should detect and-ones regardless of whether
    qualifiers are strings, lists, or missing (None / NaN / pd.NA).
    """
    qualifiers = pd.Series(
        [
            "and1",                      # simple string
            ["AND-ONE", "second_chance"],# list of strings
            None,                        # None -> False
            np.nan,                      # NaN -> False
            "no special tag",            # no match
        ]
    )

    result = box_glossary._vectorized_is_and_one(qualifiers)

    # First two rows have an and-one signal, rest do not.
    assert result.tolist() == [True, True, False, False, False]


def test_accumulate_player_counts_tracks_glossary_free_throw_trips():
    pbp = box_glossary.annotate_events(_mini_true_shooting_pbp())

    counts = box_glossary.accumulate_player_counts(pbp)

    player_counts = counts[counts["player_id"] == 10].iloc[0]

    assert player_counts["technical_free_throw_trips"] == 1
    assert player_counts["x2pt_and_1_free_throw_trips"] == 1
    assert player_counts.get("x3pt_and_1_free_throw_trips", 0) == 0
    assert player_counts["FTA"] == 2


def test_build_player_box_uses_glossary_true_shooting_inputs():
    pbp = box_glossary.annotate_events(_mini_true_shooting_pbp())
    counts = box_glossary.accumulate_player_counts(pbp)

    exposures = pd.DataFrame(
        [
            {
                "game_id": 1,
                "team_id": 1,
                "player_id": 10,
                "Minutes": 10.0,
                "POSS_OFF": 10,
                "POSS_DEF": 10,
                "OnCourt_For_OREB_Total": 0,
                "OnCourt_For_DREB_Total": 0,
                "OnCourt_Opp_2p_Att": 0,
                "OnCourt_Team_FGM": 0,
                "OnCourt_Team_FGA": 0,
                "OnCourt_Team_Points": 0,
                "OnCourt_Opp_Points": 0,
                "OnCourt_Team_FT_Att": 0,
                "OnCourt_Team_FT_Made": 0,
                "OnCourt_Team_3p_Att": 0,
                "OnCourt_Team_3p_Made": 0,
            }
        ]
    )

    box = box_glossary.build_player_box(pbp, counts, exposures)

    player_row = box[box["player_id"] == 10].iloc[0]

    assert np.isclose(player_row["TSAttempts"], 1.0)
    assert np.isclose(player_row["TSPoss"], player_row["TSAttempts"])
    assert np.isclose(player_row["PossessionsUsed"], player_row["TSAttempts"] + player_row["TOV"])
    assert np.isclose(player_row["TSpct"], player_row["PTS"] / (2 * player_row["TSAttempts"]))
    assert np.isclose(player_row["USG"], player_row["PossessionsUsed"] / player_row["POSS_OFF"])


def test_annotate_events_goaltend_via_qualifiers():
    """
    annotate_events should mark is_goaltend when 'goaltend' appears in the
    qualifiers column, even if subfamily text doesn't include it.
    """
    df = pd.DataFrame(
        {
            "game_id": [1, 1],
            "home_team_id": [100, 100],
            "away_team_id": [200, 200],
            "home_team_abbrev": ["HOM", "HOM"],
            "away_team_abbrev": ["AWY", "AWY"],
            "team_id": [100, 100],
            "event_type_de": ["shot", "shot"],
            "subfamily_de": ["layup", "layup"],
            "qualifiers": [
                ["goaltend"],   # should be flagged
                ["something"],  # should not
            ],
            "points_made": [0, 0],
        }
    )

    annotated = box_glossary.annotate_events(df)

    assert annotated["is_goaltend"].tolist() == [True, False]


def test_accumulate_player_counts_assist_fallback_same_team_only():
    """
    The V2 fallback assist logic should only credit assists to player2_id
    when player2 is on the same team as the shooter.
    """
    df = pd.DataFrame(
        [
            # Teammate assist via player2_id
            {
                "game_id": 1,
                "player1_id": 10,           # shooter
                "player1_team_id": 1,
                "player2_id": 20,           # teammate
                "player2_team_id": 1,
                "assist_id": None,
                "is_fg_attempt": True,
                "is_fg_make": True,
                "is_three": False,
                "shot_zone": "0_3",
                "points_made": 2,
                "family": "shot",
            },
            # Opponent as player2_id: should NOT be credited with AST
            {
                "game_id": 1,
                "player1_id": 30,           # shooter
                "player1_team_id": 1,
                "player2_id": 40,           # opponent
                "player2_team_id": 2,
                "assist_id": None,
                "is_fg_attempt": True,
                "is_fg_make": True,
                "is_three": False,
                "shot_zone": "0_3",
                "points_made": 2,
                "family": "shot",
            },
        ]
    )

    counts = box_glossary.accumulate_player_counts(df)

    # Index by player_id for easier assertions
    by_player = {
        (row["team_id"], row["player_id"]): row for _, row in counts.iterrows()
    }

    # Shooter 10: made FG, assisted, should have FGM, FGM_AST, and zonal stats
    p10 = by_player[(1, 10)]
    assert p10["FGA"] == 1
    assert p10["FGM"] == 1
    assert p10["FGM_AST"] == 1
    assert p10["0_3_FGA"] == 1
    assert p10["0_3_FGM"] == 1
    assert p10["0_3_FGM_AST"] == 1

    # Teammate 20: credited with one AST, and zonal AST_0_3
    p20 = by_player[(1, 20)]
    assert p20["AST"] == 1
    assert p20["AST_0_3"] == 1

    # Shooter 30: made shot but no valid teammate assist in the data -> unassisted
    p30 = by_player[(1, 30)]
    assert p30["FGA"] == 1
    assert p30["FGM"] == 1
    # No assisted makes for this shooter
    assert p30.get("FGM_AST", 0) == 0
    # Unassisted counters should pick this up
    assert p30["FGA_UNAST"] == 1
    assert p30["FGM_UNAST"] == 1

    # Opponent 40 should NOT appear with an AST entry at all
    assert (1, 40) not in by_player


def test_build_player_box_includes_dnp_players_when_meta_provided():
    """
    When player_game_meta is provided (and outer-merged), build_player_box should
    include DNP players even if they have 0 minutes and no on-court exposure.
    """
    # Minimal pbp frame for team mapping
    pbp_df = pd.DataFrame(
        {
            "game_id": [1],
            "home_team_id": [1],
            "away_team_id": [2],
            "home_team_abbrev": ["HOM"],
            "away_team_abbrev": ["AWY"],
        }
    )

    # Counts for two players who actually played
    counts_df = pd.DataFrame(
        [
            {
                "game_id": 1,
                "team_id": 1,
                "player_id": 101,
                "FGA": 10,
                "FGM": 5,
                "FTA": 4,
                "FTM": 3,
                "PTS": 13,
            },
            {
                "game_id": 1,
                "team_id": 1,
                "player_id": 102,
                "FGA": 5,
                "FGM": 2,
                "FTA": 2,
                "FTM": 2,
                "PTS": 6,
            },
        ]
    )

    # Exposures for those two players
    exposures_df = pd.DataFrame(
        [
            {
                "game_id": 1,
                "team_id": 1,
                "player_id": 101,
                "Minutes": 30.0,
                "POSS_OFF": 50,
                "POSS_DEF": 50,
                "OnCourt_Team_Points": 60,
                "OnCourt_Opp_Points": 55,
                "OnCourt_Team_FGM": 25,
                "OnCourt_For_OREB_Total": 20,
                "OnCourt_For_DREB_Total": 25,
                "OnCourt_Opp_2p_Att": 30,
            },
            {
                "game_id": 1,
                "team_id": 1,
                "player_id": 102,
                "Minutes": 18.0,
                "POSS_OFF": 25,
                "POSS_DEF": 25,
                "OnCourt_Team_Points": 30,
                "OnCourt_Opp_Points": 28,
                "OnCourt_Team_FGM": 12,
                "OnCourt_For_OREB_Total": 10,
                "OnCourt_For_DREB_Total": 12,
                "OnCourt_Opp_2p_Att": 15,
            },
        ]
    )

    # Player game meta includes a third player (DNP)
    player_game_meta = pd.DataFrame(
        [
            {"game_id": 1, "team_id": 1, "player_id": 101, "DNP": 0, "Inactive": 0},
            {"game_id": 1, "team_id": 1, "player_id": 102, "DNP": 0, "Inactive": 0},
            {"game_id": 1, "team_id": 1, "player_id": 103, "DNP": 1, "Inactive": 0},
        ]
    )

    box = box_glossary.build_player_box(
        pbp_df,
        counts_df,
        exposures_df,
        player_meta=None,
        game_meta=None,
        player_game_meta=player_game_meta,
    )

    # Filter to team 1 for clarity
    team_rows = box[box["team_id"] == 1]

    # All three players (101, 102, 103) should be present
    player_ids = sorted(team_rows["player_id"].unique().tolist())
    assert player_ids == [101, 102, 103]

    # DNP player should have 0 minutes but DNP flag set
    dnp_row = team_rows[team_rows["player_id"] == 103].iloc[0]
    assert dnp_row["Minutes"] == 0
    assert dnp_row["DNP"] == 1