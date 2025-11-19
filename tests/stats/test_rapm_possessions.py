from pathlib import Path
import sys

import pytest

pytest.importorskip("pandas")
import pandas as pd  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parents[2]))

from nba_scraper.stats import PbP  # noqa: E402

FIXTURES = Path(__file__).resolve().parents[1] / "test_files"


def test_rapm_possessions_from_csv_games():
    csv_games = sorted(FIXTURES.glob("*.csv"))
    assert csv_games, "No play-by-play CSV fixtures found"

    all_possessions = []
    for csv_path in csv_games:
        pbp_df = pd.read_csv(csv_path)
        pbp = PbP(pbp_df)
        poss_df = pbp.rapm_possessions()

        assert not poss_df.empty
        all_possessions.append(poss_df)

    combined = pd.concat(all_possessions, ignore_index=True)
    assert not combined.empty
