import copy
import importlib
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("pandas")

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from nba_scraper import cdn_parser, io_sources, lineup_builder, v2_parser
from nba_scraper.mapping import event_codebook, normalize_descriptor
from nba_scraper.parser_utils import infer_possession_after

FIXTURES = Path(__file__).parent / "test_files"


def _load_json(name: str):
    with (FIXTURES / name).open(encoding="utf-8") as fh:
        return json.load(fh)


def _build_lineup_df(rows, *, home_id=1, away_id=2):
    base = {
        "game_id": "test_game",
        "period": 1,
        "seconds_elapsed": 0,
        "family": "other",
        "subfamily": "",
        "player1_id": None,
        "player2_id": None,
        "player3_id": None,
        "player1_team_id": home_id,
        "team_id": home_id,
        "event_type_de": "live",
        "home_team_id": home_id,
        "away_team_id": away_id,
        "player1_name": "",
        "player2_name": "",
        "player3_name": "",
    }
    data = []
    for idx, row in enumerate(rows, start=1):
        record = base.copy()
        record["eventnum"] = idx
        record.update(row)
        data.append(record)
    return pd.DataFrame(data)


def test_cdn_parse_basic():
    pbp = _load_json("cdn_playbyplay_0022400001.json")
    box = _load_json("cdn_boxscore_0022400001.json")
    df = cdn_parser.parse_actions_to_rows(pbp, box)
    df = lineup_builder.attach_lineups(df, box_json=box, pbp_json=pbp)
    assert not df.empty
    assert "home_player_1_id" in df.columns
    assert {"eventnum", "season"}.issubset(df.columns)
    expected_columns = {
        "game_id",
        "period",
        "pctimestring",
        "seconds_elapsed",
        "eventmsgtype",
        "eventmsgactiontype",
        "event_type_de",
        "event_team",
        "player1_team_id",
        "player2_team_id",
        "player3_team_id",
        "points_made",
        "steal_id",
        "block_id",
        "is_turnover",
        "is_steal",
        "is_block",
        "ft_n",
        "ft_m",
        "season",
        "home_player_1",
        "away_player_1",
    }
    assert expected_columns.issubset(df.columns)
    turnover_row = df[df["family"] == "turnover"].iloc[0]
    assert turnover_row["steal_id"] == 201939
    assert turnover_row["is_turnover"] == 1
    assert turnover_row["is_steal"] == 1
    shot_row = df[df["family"] == "2pt"].iloc[0]
    assert shot_row["block_id"] == 201143
    assert shot_row["is_block"] == 1


def test_cdn_game_date_normalized():
    pbp = _load_json("cdn_playbyplay_0022400001.json")
    box = _load_json("cdn_boxscore_0022400001.json")
    df = cdn_parser.parse_actions_to_rows(pbp, box)
    val = str(df["game_date"].iloc[0])
    assert len(val) == 10


def test_eventnum_is_numeric():
    v2 = _load_json("v2_pbp_0021700001.json")
    df2 = v2_parser.parse_v2_to_rows(v2)
    assert "eventnum" in df2.columns
    assert df2["eventnum"].dtype.kind in ("i", "u")

    pbp = _load_json("cdn_playbyplay_0022400001.json")
    box = _load_json("cdn_boxscore_0022400001.json")
    dfc = cdn_parser.parse_actions_to_rows(pbp, box)
    assert "eventnum" in dfc.columns
    assert dfc["eventnum"].dtype.kind in ("i", "u")


def test_cdn_and_v2_share_canonical_columns():
    cdn_df = io_sources.parse_any(
        (
            FIXTURES / "cdn_playbyplay_0022400001.json",
            FIXTURES / "cdn_boxscore_0022400001.json",
        ),
        io_sources.SourceKind.CDN_LOCAL,
    )
    v2_df = io_sources.parse_any(
        FIXTURES / "v2_pbp_0021700001.json",
        io_sources.SourceKind.V2_LOCAL,
    )

    assert list(cdn_df.columns) == list(v2_df.columns)


def test_v2_parse_basic():
    v2 = _load_json("v2_pbp_0021700001.json")
    df = v2_parser.parse_v2_to_rows(v2)
    assert not df.empty
    assert {"eventnum", "season"}.issubset(df.columns)
    assert set(
        [
            "family",
            "eventmsgtype",
            "points_made",
            "event_type_de",
            "event_team",
            "player1_team_id",
            "season",
        ]
    ).issubset(df.columns)
    made = df[df["eventmsgtype"] == 1].iloc[0]
    assert made["points_made"] == 2
    assert made["event_type_de"] == "shot"


def test_v2_game_date_normalized():
    v2 = _load_json("v2_pbp_0021700001.json")
    df = v2_parser.parse_v2_to_rows(v2)
    val = str(df["game_date"].iloc[0])
    assert len(val) == 10


def test_seconds_elapsed():
    pbp = _load_json("cdn_playbyplay_0022400001.json")
    box = _load_json("cdn_boxscore_0022400001.json")
    df = cdn_parser.parse_actions_to_rows(pbp, box)
    first = df.iloc[0]
    assert first["seconds_elapsed"] == 0
    shot = df[df["family"].isin({"2pt", "3pt"})].iloc[0]
    assert shot["seconds_elapsed"] > first["seconds_elapsed"]


def test_points_made():
    pbp = _load_json("cdn_playbyplay_0022400001.json")
    box = _load_json("cdn_boxscore_0022400001.json")
    df = cdn_parser.parse_actions_to_rows(pbp, box)
    three = df[(df["family"] == "3pt") & (df["shot_made"] == 1)].iloc[0]
    assert three["points_made"] == 3
    layup = df[(df["family"] == "2pt") & (df["shot_made"] == 0)].iloc[0]
    assert layup["points_made"] == 0


def test_sidecar_merge():
    pbp = _load_json("cdn_playbyplay_0022400001.json")
    box = _load_json("cdn_boxscore_0022400001.json")
    df = cdn_parser.parse_actions_to_rows(pbp, box)
    assert "steal" not in df["family"].values
    assert "block" not in df["family"].values


def test_cdn_style_flags_exclude_control_descriptors():
    pbp = {
        "game": {
            "gameId": "test",
            "actions": [
                {
                    "actionNumber": 1,
                    "orderNumber": 1,
                    "period": 1,
                    "clock": "PT12M00.00S",
                    "actionType": "jumpball",
                    "subType": "center",
                    "descriptor": "StartPeriod",
                    "qualifiers": ["startperiod"],
                },
                {
                    "actionNumber": 2,
                    "orderNumber": 2,
                    "period": 1,
                    "clock": "PT11M59.00S",
                    "actionType": "2pt",
                    "subType": "layup",
                    "descriptor": "Challenge",
                    "qualifiers": ["challenge"],
                    "teamId": 1,
                    "teamTricode": "HOM",
                    "personId": 10,
                    "playerName": "Test Shooter",
                    "shotResult": "Missed",
                    "shotActionNumber": 1,
                    "score": {"home": 0, "away": 0},
                },
            ],
        }
    }
    box = {
        "game": {
            "gameId": "test",
            "gameTimeUTC": "2024-01-01T00:00:00Z",
            "homeTeam": {
                "teamId": 1,
                "teamTricode": "HOM",
                "players": [{"personId": 10, "status": "ACTIVE"}],
            },
            "awayTeam": {
                "teamId": 2,
                "teamTricode": "AWY",
                "players": [{"personId": 20, "status": "ACTIVE"}],
            },
        }
    }

    df = cdn_parser.parse_actions_to_rows(pbp, box)
    jump_row = df[df["family"] == "jumpball"].iloc[0]
    assert "startperiod" not in (jump_row["style_flags"] or [])

    challenge_row = df[df["qualifiers"].apply(lambda q: "challenge" in (q or []))].iloc[0]
    assert "challenge" in challenge_row["qualifiers"]
    assert "challenge" not in (challenge_row["style_flags"] or [])


def test_event_codebook_maps_free_throw_actiontypes():
    cases = [
        ("Free Throw 1 of 1", 10),
        ("Free Throw 1 of 2", 11),
        ("Free Throw 2 of 2", 12),
        ("Free Throw 1 of 3", 13),
        ("Free Throw 2 of 3", 14),
        ("Free Throw 3 of 3", 15),
    ]
    for label, expected in cases:
        assert event_codebook.actiontype_code_for("freethrow", label) == expected


def test_infer_possession_after_respects_feed_on_final_ft():
    home = 1610612737
    away = 1610612744
    df = pd.DataFrame(
        [
            {
                "game_id": "test",
                "period": 1,
                "event_type_de": "freethrow",
                "home_team_id": home,
                "away_team_id": away,
                "team_id": home,
                "eventmsgtype": 3,
                "family": "freethrow",
                "shot_made": 1,
                "ft_n": 1,
                "ft_m": 1,
                "is_d_rebound": 0,
                "possession_after": home,
            }
        ]
    )

    result = infer_possession_after(df)
    assert result.loc[0, "possession_after"] == home


def test_substitution_queue_scoped_to_clock_tick():
    game_id = "test_game"
    home_id = 1
    away_id = 2
    starters = {
        "home": [10, 11, 12, 13, 14],
        "away": [20, 21, 22, 23, 24],
    }
    rows = [
        {
            "game_id": game_id,
            "home_team_id": home_id,
            "away_team_id": away_id,
            "family": "jumpball",
            "seconds_elapsed": 0,
            "player1_id": 10,
            "player1_team_id": home_id,
            "team_id": home_id,
        },
        {
            "game_id": game_id,
            "home_team_id": home_id,
            "away_team_id": away_id,
            "family": "substitution",
            "subfamily": "out",
            "seconds_elapsed": 100,
            "player1_id": 14,
            "player1_team_id": home_id,
            "team_id": home_id,
        },
        {
            "game_id": game_id,
            "home_team_id": home_id,
            "away_team_id": away_id,
            "family": "substitution",
            "subfamily": "in",
            "seconds_elapsed": 200,
            "player1_id": 15,
            "player1_team_id": home_id,
            "team_id": home_id,
        },
    ]
    df = pd.DataFrame(rows)
    result = lineup_builder.attach_lineups(df, starters=starters)

    cols = [f"home_player_{idx}_id" for idx in range(1, 6)]
    initial = tuple(int(result.loc[0, col]) for col in cols)
    after_out = tuple(int(result.loc[1, col]) for col in cols)
    after_in = tuple(int(result.loc[2, col]) for col in cols)

    assert initial == after_out
    assert initial == after_in


def test_duplicate_sub_in_is_noop():
    lineup = [1, 2, 3, 4, 5]
    lineup_builder._apply_substitution(lineup, sub_out=6, sub_in=3)

    assert lineup == [1, 2, 3, 4, 5]
    assert len({pid for pid in lineup if pid is not None}) == 5


def test_cdn_parser_populates_secondary_names_from_json():
    pbp = _load_json("cdn_playbyplay_0022400001.json")
    box = _load_json("cdn_boxscore_0022400001.json")
    df = cdn_parser.parse_actions_to_rows(pbp, box)

    assist_row = df[(df["assist_id"] != 0) & (df["shot_made"] == 1)].iloc[0]
    assert assist_row["player2_name"] == "Stephen Curry"

    steal_row = df[df["steal_id"] != 0].iloc[0]
    assert steal_row["player2_name"] == "Stephen Curry"

    block_row = df[df["block_id"] != 0].iloc[0]
    assert block_row["player3_name"] == "Paul Millsap"


def test_cdn_scores_populated():
    pbp = _load_json("cdn_playbyplay_0022400001.json")
    box = _load_json("cdn_boxscore_0022400001.json")
    df = cdn_parser.parse_actions_to_rows(pbp, box)
    assert df["score_home"].notna().any()
    assert df["score_away"].notna().any()
    assert df["scoremargin"].str.len().max() > 0


def test_cdn_synth_ft_description(monkeypatch):
    monkeypatch.setenv("NBA_SCRAPER_SYNTH_FT_DESC", "1")
    # Reload module to pick up environment variable change.
    module = importlib.reload(cdn_parser)
    globals()["cdn_parser"] = module
    pbp = _load_json("cdn_playbyplay_0022400001.json")
    actions = pbp["game"].get("actions", [])
    if not any("of" in (act.get("subType") or "").lower() for act in actions):
        pbp = copy.deepcopy(pbp)
        actions = pbp["game"].setdefault("actions", [])
        actions.append(
            {
                "actionNumber": 11,
                "orderNumber": 11,
                "period": 1,
                "clock": "PT08M30.00S",
                "actionType": "freethrow",
                "subType": "Free Throw 1 of 2",
                "teamId": 1610612747,
                "teamTricode": "LAL",
                "personId": 2544,
                "playerName": "LeBron James",
                "shotResult": "Made",
                "shotActionNumber": 4,
                "descriptor": "shooting foul",
                "score": {"home": 4, "away": 3},
            }
        )
    box = _load_json("cdn_boxscore_0022400001.json")
    df = module.parse_actions_to_rows(pbp, box)
    ft_rows = df[df["family"] == "freethrow"]
    assert not ft_rows.empty
    has_desc = ft_rows[
        (ft_rows["homedescription"] != "") | (ft_rows["visitordescription"] != "")
    ]
    assert not has_desc.empty
    # Reset module state for other tests
    monkeypatch.setenv("NBA_SCRAPER_SYNTH_FT_DESC", "0")
    module = importlib.reload(module)
    globals()["cdn_parser"] = module


def test_descriptor_normalization_extracts_core_and_styles():
    core, styles = normalize_descriptor("Driving Floating Bank Layup")
    assert core == "layup"
    assert set(styles) == {"driving", "floating", "bank"}


def test_descriptor_normalization_handles_alley_oop_variants():
    for variant in ("Alley Oop Dunk", "alley-oop dunk"):
        core, styles = normalize_descriptor(variant)
        assert core == "dunk"
        assert styles == ["alleyoop"]


def test_lineups_have_five_players_every_live_row():
    pbp = _load_json("cdn_playbyplay_0022400001.json")
    box = _load_json("cdn_boxscore_0022400001.json")
    df = io_sources.parse_any((pbp, box), io_sources.SourceKind.CDN_LOCAL)
    df = lineup_builder.attach_lineups(df, box_json=box, pbp_json=pbp)

    live = ~df["event_type_de"].isin(["period", "timeout"])
    for side in ("home", "away"):
        cols = [f"{side}_player_{i}_id" for i in range(1, 6)]
        numeric = (
            df.loc[live, cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .astype(int)
        )
        assert ((numeric > 0).sum(axis=1) == 5).all()


def test_lineups_have_five_unique_players():
    pbp = _load_json("cdn_playbyplay_0022400001.json")
    box = _load_json("cdn_boxscore_0022400001.json")
    df = io_sources.parse_any((pbp, box), io_sources.SourceKind.CDN_LOCAL)
    df = lineup_builder.attach_lineups(df, box_json=box, pbp_json=pbp)

    live = ~df["event_type_de"].isin(["period", "timeout"])
    for side in ("home", "away"):
        cols = [f"{side}_player_{i}_id" for i in range(1, 6)]
        lineups = df.loc[live, cols].astype(int)
        assert lineups.apply(lambda r: len(set(r)) == 5, axis=1).all()


def test_cdn_split_substitution_updates_lineups():
    pbp = _load_json("cdn_playbyplay_0022400001.json")
    actions = pbp["game"].get("actions", [])
    needs_split = not any(
        act.get("actionType") == "substitution"
        and (act.get("subType") or "").lower() == "out"
        for act in actions
    )
    if needs_split:
        pbp = copy.deepcopy(pbp)
        actions = pbp["game"].setdefault("actions", [])
        actions.extend(
            [
                {
                    "actionNumber": 21,
                    "orderNumber": 21,
                    "period": 1,
                    "clock": "PT08M45.00S",
                    "actionType": "substitution",
                    "subType": "out",
                    "teamId": 1610612747,
                    "teamTricode": "LAL",
                    "personId": 2732,
                    "playerName": "Patrick Beverley",
                    "shotActionNumber": 5,
                },
                {
                    "actionNumber": 22,
                    "orderNumber": 22,
                    "period": 1,
                    "clock": "PT08M45.00S",
                    "actionType": "substitution",
                    "subType": "in",
                    "teamId": 1610612747,
                    "teamTricode": "LAL",
                    "personId": 203210,
                    "playerName": "Kentavious Caldwell-Pope",
                    "shotActionNumber": 5,
                },
            ]
        )
    box = _load_json("cdn_boxscore_0022400001.json")
    df = io_sources.parse_any((pbp, box), io_sources.SourceKind.CDN_LOCAL)
    df = lineup_builder.attach_lineups(df, box_json=box, pbp_json=pbp)

    sub_in = df[
        (df["family"] == "substitution")
        & (df["subfamily"] == "in")
        & (df["player1_id"] == 203210)
    ].iloc[0]
    idx = sub_in.name

    home_ids = [df.loc[idx, f"home_player_{i}_id"] for i in range(1, 6)]
    assert 203210 in home_ids


def test_team_fields_filled_on_team_events():
    pbp = _load_json("cdn_playbyplay_0022400001.json")
    box = _load_json("cdn_boxscore_0022400001.json")
    df = io_sources.parse_any((pbp, box), io_sources.SourceKind.CDN_LOCAL)

    team_events = ~df["event_type_de"].isin(["period", "timeout", "jump-ball"])
    team_ids = (
        pd.to_numeric(df.loc[team_events, "team_id"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    assert (team_ids != 0).all()
    tricodes = df.loc[team_events, "team_tricode"].fillna("")
    assert (tricodes != "").all()


def test_shots_have_xy_or_are_synthesized():
    pbp = _load_json("cdn_playbyplay_0022400001.json")
    box = _load_json("cdn_boxscore_0022400001.json")
    df = io_sources.parse_any((pbp, box), io_sources.SourceKind.CDN_LOCAL)

    shots = df["family"].isin(["2pt", "3pt"])
    xy_present = shots & df["x"].notna() & df["y"].notna()

    def _has_flag(flags):
        if isinstance(flags, (list, tuple, set)):
            return "xy_synth" in flags
        return False

    synth = shots & df["style_flags"].apply(_has_flag)
    assert (xy_present | synth)[shots].all()


def test_shotchart_backfill_overrides_synth(monkeypatch):
    pbp = FIXTURES / "cdn_playbyplay_0022400001.json"
    box = FIXTURES / "cdn_boxscore_0022400001.json"
    monkeypatch.setenv("NBA_SCRAPER_BACKFILL_COORDS", "1")
    monkeypatch.setattr(io_sources, "_BACKFILL_COORDS", True)

    df = io_sources.parse_any((pbp, box), io_sources.SourceKind.CDN_LOCAL)
    expected = {
        3: (5.5, 71.25),
        7: (83.75, 12.5),
    }
    rows = df[df["eventnum"].isin(expected)].set_index("eventnum")
    assert len(rows) == len(expected)
    for eventnum, (exp_x, exp_y) in expected.items():
        row = rows.loc[eventnum]
        assert row["x"] == pytest.approx(exp_x)
        assert row["y"] == pytest.approx(exp_y)
        assert "xy_synth" not in (row["style_flags"] or [])


def test_lineup_builder_substitution_uses_player1_team_id_when_team_missing():
    home_id = 1610612737
    away_id = 1610612744
    df = pd.DataFrame(
        [
            {
                "game_id": "test-game",
                "period": 1,
                "family": "rebound",
                "team_id": home_id,
                "player1_id": 2001,
                "player1_team_id": home_id,
                "player2_id": 0,
                "player3_id": 0,
                "player2_team_id": 0,
                "player3_team_id": 0,
                "player1_name": "",
                "player2_name": "",
                "player3_name": "",
                "home_team_id": home_id,
                "away_team_id": away_id,
            },
            {
                "game_id": "test-game",
                "period": 1,
                "family": "substitution",
                "team_id": 0,
                "player1_id": 2001,
                "player2_id": 2002,
                "player1_team_id": home_id,
                "player2_team_id": 0,
                "player3_id": 0,
                "player3_team_id": 0,
                "player1_name": "",
                "player2_name": "",
                "player3_name": "",
                "home_team_id": home_id,
                "away_team_id": away_id,
            },
        ]
    )

    result = lineup_builder.attach_lineups(df)

    assert result.loc[0, "home_player_1_id"] == 2001
    assert result.loc[1, "home_player_1_id"] == 2002


def test_lineup_builder_handles_batched_substitutions_single_tick():
    starters = {"home": [1, 2, 3, 4, 5], "away": [101, 102, 103, 104, 105]}
    rows = [
        {"family": "jump-ball", "seconds_elapsed": 0},
        *[
            {
                "family": "substitution",
                "subfamily": "out",
                "seconds_elapsed": 30,
                "player1_id": pid,
            }
            for pid in (1, 2, 3)
        ],
        *[
            {
                "family": "substitution",
                "subfamily": "in",
                "seconds_elapsed": 30,
                "player1_id": pid,
            }
            for pid in (6, 7, 8)
        ],
    ]
    df = _build_lineup_df(rows)
    result = lineup_builder.attach_lineups(df, starters=starters)

    final = [int(result.iloc[-1][f"home_player_{i}_id"]) for i in range(1, 6)]
    assert set(final) == {4, 5, 6, 7, 8}


def test_lineup_builder_fills_orphan_sub_in_into_empty_slot():
    rows = [
        {"family": "rebound", "player1_id": pid}
        for pid in (1, 2, 3, 4)
    ]
    rows.append(
        {
            "family": "substitution",
            "subfamily": "in",
            "player1_id": 5,
        }
    )
    df = _build_lineup_df(rows)
    result = lineup_builder.attach_lineups(df)

    last = [int(result.iloc[-1][f"home_player_{i}_id"]) for i in range(1, 6)]
    assert set(last) == {1, 2, 3, 4, 5}


def test_lineup_builder_uses_player1_team_when_team_id_zero():
    starters = {"home": [1, 2, 3, 4, 5], "away": [101, 102, 103, 104, 105]}
    rows = [
        {"family": "rebound", "player1_id": 1},
        {
            "family": "substitution",
            "subfamily": "out",
            "player1_id": 1,
            "team_id": 0,
        },
        {
            "family": "substitution",
            "subfamily": "in",
            "player1_id": 6,
            "team_id": 0,
        },
    ]
    df = _build_lineup_df(rows)
    result = lineup_builder.attach_lineups(df, starters=starters)

    last = [int(result.iloc[-1][f"home_player_{i}_id"]) for i in range(1, 6)]
    assert 6 in last
    assert 1 not in last


def test_possession_after_filled_between_anchors():
    pbp = _load_json("cdn_playbyplay_0022400001.json")
    box = _load_json("cdn_boxscore_0022400001.json")
    df = io_sources.parse_any((pbp, box), io_sources.SourceKind.CDN_LOCAL)
    df = infer_possession_after(df)

    by_period = df.groupby(["game_id", "period"])
    live = ~df["event_type_de"].isin(["period", "timeout"])
    mask = live & df["possession_after"].notna()
    checks = []
    for _, group in by_period:
        live_mask = live.loc[group.index]
        live_vals = mask.loc[group.index][live_mask]
        if live_vals.empty:
            checks.append(True)
            continue
        checks.append(live_vals.iloc[1:].all())
    assert all(checks)


def test_possession_inference_handles_mixed_sequence():
    home = 100
    away = 200
    rows = [
        {
            "family": "2pt",
            "team_id": home,
            "shot_made": 1,
        },
        {
            "family": "rebound",
            "team_id": away,
            "is_d_rebound": 1,
        },
        {
            "family": "turnover",
            "eventmsgtype": 5,
            "team_id": away,
        },
        {
            "family": "3pt",
            "team_id": home,
            "shot_made": 1,
        },
        {
            "family": "freethrow",
            "team_id": home,
            "shot_made": 1,
            "ft_n": 1,
            "ft_m": 2,
        },
        {
            "family": "freethrow",
            "team_id": home,
            "shot_made": 1,
            "ft_n": 2,
            "ft_m": 2,
        },
        {
            "family": "freethrow",
            "team_id": home,
            "shot_made": 1,
            "ft_n": 1,
            "ft_m": 1,
            "possession_after": home,
        },
    ]
    df = _build_lineup_df(rows, home_id=home, away_id=away)
    df["possession_after"] = df.get("possession_after", 0).fillna(0)
    df = infer_possession_after(df)

    expected = [away, away, home, away, away, away, home]
    assert list(df["possession_after"]) == expected
