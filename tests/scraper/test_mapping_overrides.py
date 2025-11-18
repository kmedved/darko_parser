import importlib
import json
from pathlib import Path

import pytest

pytest.importorskip("pandas")

from nba_scraper import cdn_parser, v2_parser

FIX = Path(__file__).parent / "test_files"
MAP = FIX / "mapping_min.yml"


def _load_json(name: str):
    with open(FIX / name, "r", encoding="utf-8") as handle:
        return json.load(handle)


def test_cdn_yaml_override_env(monkeypatch):
    monkeypatch.setenv("NBA_SCRAPER_MAP", str(MAP))
    module = importlib.reload(cdn_parser)
    globals()["cdn_parser"] = module
    pbp = _load_json("cdn_playbyplay_0022400001.json")
    box = _load_json("cdn_boxscore_0022400001.json")
    df = module.parse_actions_to_rows(pbp, box)
    row = df[df["family"] == "turnover"].iloc[0]
    assert row["eventmsgactiontype"] == 2
    assert row["subfamily"] == "lost ball"
    monkeypatch.delenv("NBA_SCRAPER_MAP", raising=False)
    module = importlib.reload(module)
    globals()["cdn_parser"] = module


def test_v2_yaml_override_env(monkeypatch):
    monkeypatch.setenv("NBA_SCRAPER_MAP", str(MAP))
    module = importlib.reload(v2_parser)
    globals()["v2_parser"] = module
    v2_json = _load_json("v2_pbp_0021700001.json")
    df = module.parse_v2_to_rows(v2_json)
    row = df[df["family"] == "turnover"].iloc[0]
    assert row["eventmsgactiontype"] == 2
    assert row["subfamily"] == "lost ball"
    monkeypatch.delenv("NBA_SCRAPER_MAP", raising=False)
    module = importlib.reload(module)
    globals()["v2_parser"] = module
