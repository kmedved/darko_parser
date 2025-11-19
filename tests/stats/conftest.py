from __future__ import annotations

from pathlib import Path
from typing import Callable, Union

import pandas as pd
import pytest

from nba_scraper.stats import PbP

PathLike = Union[str, Path]

STATS_FIXTURES = Path(__file__).resolve().parent / "test_files"
ID_COLUMNS: tuple[str, ...] = (
    "game_id",
    "team_id",
    "home_team_id",
    "away_team_id",
    "player1_team_id",
    "player2_team_id",
    "player3_team_id",
    *(f"home_player_{i}_id" for i in range(1, 6)),
    *(f"away_player_{i}_id" for i in range(1, 6)),
)


def _resolve_stats_path(pathlike: PathLike) -> Path:
    path = Path(pathlike)
    if not path.is_absolute():
        path = STATS_FIXTURES / path
    return path


def _coerce_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in ID_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    if "season" in df.columns:
        df["season"] = pd.to_numeric(df["season"], errors="coerce").fillna(0).astype(int)
    return df


def _load_stats_csv(pathlike: PathLike) -> pd.DataFrame:
    df = pd.read_csv(_resolve_stats_path(pathlike))
    return _coerce_identifier_columns(df)


@pytest.fixture(scope="session")
def stats_fixture_dir() -> Path:
    return STATS_FIXTURES


@pytest.fixture(scope="session")
def stats_csv_loader() -> Callable[[PathLike], pd.DataFrame]:
    def _loader(pathlike: PathLike, *, season: int | None = None) -> pd.DataFrame:
        df = _load_stats_csv(pathlike)
        if season is not None:
            df["season"] = season
        return df

    return _loader


@pytest.fixture(scope="session")
def v2_pbp_path() -> Path:
    return STATS_FIXTURES / "20700233.csv"


@pytest.fixture(scope="session")
def v2_alt_pbp_path() -> Path:
    return STATS_FIXTURES / "21100736.csv"


@pytest.fixture(scope="session")
def cdn_0021900151_path() -> Path:
    return STATS_FIXTURES / "0021900151_cdn.csv"


@pytest.fixture(scope="session")
def _v2_pbp_df_base(stats_csv_loader, v2_pbp_path) -> pd.DataFrame:
    return stats_csv_loader(v2_pbp_path, season=2008)


@pytest.fixture(scope="session")
def _v2_alt_pbp_df_base(stats_csv_loader, v2_alt_pbp_path) -> pd.DataFrame:
    return stats_csv_loader(v2_alt_pbp_path, season=2012)


@pytest.fixture(scope="session")
def _cdn_0021900151_df_base(stats_csv_loader, cdn_0021900151_path) -> pd.DataFrame:
    return stats_csv_loader(cdn_0021900151_path, season=2020)


@pytest.fixture
def v2_pbp_df(_v2_pbp_df_base) -> pd.DataFrame:
    return _v2_pbp_df_base.copy()


@pytest.fixture
def v2_alt_pbp_df(_v2_alt_pbp_df_base) -> pd.DataFrame:
    return _v2_alt_pbp_df_base.copy()


@pytest.fixture
def cdn_0021900151_df(_cdn_0021900151_df_base) -> pd.DataFrame:
    return _cdn_0021900151_df_base.copy()


@pytest.fixture
def pbp_v2_game(v2_pbp_df) -> PbP:
    return PbP(v2_pbp_df)


@pytest.fixture
def pbp_v2_alt_game(v2_alt_pbp_df) -> PbP:
    return PbP(v2_alt_pbp_df)


@pytest.fixture
def pbp_cdn_0021900151(cdn_0021900151_df) -> PbP:
    return PbP(cdn_0021900151_df)
