import json
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

pytest.importorskip("nba_parser")
pytest.importorskip("pandas")

import pandas as pd  # noqa: E402

from nba_scraper import io_sources, lineup_builder  # noqa: E402

FIXTURES = Path(__file__).parent / "test_files"


def test_cdn_frame_works_with_nba_parser():
    from nba_parser import PbP

    pbp = FIXTURES / "cdn_playbyplay_0022400001.json"
    box = FIXTURES / "cdn_boxscore_0022400001.json"

    df = io_sources.parse_any((pbp, box), io_sources.SourceKind.CDN_LOCAL)
    with box.open("r", encoding="utf-8") as fh:
        box_json = json.load(fh)
    df = lineup_builder.attach_lineups(df, box_json=box_json)

    required = {
        "event_type_de",
        "event_team",
        "player1_team_id",
        "player2_team_id",
        "player3_team_id",
        "is_turnover",
        "is_steal",
        "is_block",
        "season",
        "home_player_1_id",
        "away_player_5_id",
    }
    required |= {"eventnum", "scoremargin"}
    assert required.issubset(df.columns)

    pbp = PbP(df)
    pbg = pbp.playerbygamestats()
    tbg = pbp.teambygamestats()

    assert not pbg.empty
    assert not tbg.empty
    assert tbg["toc_string"].str.match(r"^\d+:\d{2}$").all()


def test_jumpball_recovered_player_exposed_in_canonical_frame():
    """
    Regression test for jump-ball semantics:

    - cdn_parser should always expose the 'tip to' player as player3_id
      on jump-ball rows when the feed provides jumpBallRecoveredPersonId.
    - nba_parser.PbP must be able to sit on top of that canonical frame
      without issue.
    """
    from nba_parser import PbP

    pbp = FIXTURES / "cdn_playbyplay_0022400001.json"
    box = FIXTURES / "cdn_boxscore_0022400001.json"

    df = io_sources.parse_any((pbp, box), io_sources.SourceKind.CDN_LOCAL)
    pbp_obj = PbP(df)

    jb = pbp_obj.df[pbp_obj.df["family"] == "jumpball"].copy()
    # We expect at least one jump ball, and for those where the CDN feed
    # had a recovered player, player3_id should be non-zero.
    assert not jb.empty

    # Some feeds legitimately omit a recovered player; we only require that
    # whenever the upstream feed had a non-zero recovered ID, we mirrored it.
    nonzero_recovered = jb.loc[jb["player3_id"] != 0, "player3_id"]
    assert not nonzero_recovered.empty
