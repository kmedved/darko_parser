import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

pytest.importorskip("pandas")

import pandas as pd

from nba_scraper import nba_scraper

FIXTURES = Path(__file__).parent / "test_files"


def test_csv_write(tmp_path):
    pbp = FIXTURES / "cdn_playbyplay_0022400001.json"
    box = FIXTURES / "cdn_boxscore_0022400001.json"
    nba_scraper.scrape_from_files(str(pbp), str(box), kind="cdn_local", data_format="csv", data_dir=tmp_path)
    output = tmp_path / f"{pbp.stem}.csv"
    assert output.exists()
    df = pd.read_csv(output)
    assert not df.empty
