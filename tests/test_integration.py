import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

pytest.importorskip("pandas")

import pandas as pd

from nba_scraper import io_sources, nba_scraper

FIXTURES = Path(__file__).parent / "test_files"


def test_scrape_from_files_cdn(tmp_path):
    pbp = FIXTURES / "cdn_playbyplay_0022400001.json"
    box = FIXTURES / "cdn_boxscore_0022400001.json"
    df = nba_scraper.scrape_from_files(str(pbp), str(box), kind="cdn_local")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_scrape_from_files_v2():
    v2 = FIXTURES / "v2_pbp_0021700001.json"
    df = nba_scraper.scrape_from_files(str(v2), kind="v2_local")
    assert not df.empty


def test_scrape_game_concat(monkeypatch):
    pbp = FIXTURES / "cdn_playbyplay_0022400001.json"
    box = FIXTURES / "cdn_boxscore_0022400001.json"
    v2 = FIXTURES / "v2_pbp_0021700001.json"

    original_parse_any = io_sources.parse_any

    def fake_parse_any(ref, kind, mapping_yaml_path=None):
        if kind == io_sources.SourceKind.CDN_REMOTE:
            return original_parse_any((pbp, box), io_sources.SourceKind.CDN_LOCAL)
        if kind == io_sources.SourceKind.V2_DICT:
            return original_parse_any(v2, io_sources.SourceKind.V2_LOCAL)
        return original_parse_any(ref, kind, mapping_yaml_path)

    monkeypatch.setattr(io_sources, "parse_any", fake_parse_any)
    df = nba_scraper.scrape_game(["0022400001", "V2"], data_format="pandas")
    assert len(df) >= 1
