"""Simple HTTP client for the NBA CDN endpoints."""
from __future__ import annotations

import logging
from typing import Any, Dict

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

PBP_URL = "https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{gid}.json"
BOX_URL = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{gid}.json"
SHOTCHART_URL = (
    "https://cdn.nba.com/static/json/liveData/shotchart/shotchart_{gid}.json"
)
SCH_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"


class _RetryingSession(requests.Session):
    def __init__(self) -> None:
        super().__init__()
        retry = Retry(
            total=5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
            backoff_factor=0.5,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.mount("https://", adapter)
        self.mount("http://", adapter)


_session: _RetryingSession | None = None


def _get_session() -> _RetryingSession:
    global _session
    if _session is None:
        _session = _RetryingSession()
    return _session


def _fetch_json(url: str) -> Dict[str, Any]:
    session = _get_session()
    logger.debug("Fetching %s", url)
    response = session.get(url, timeout=10)
    response.raise_for_status()
    return response.json()


def fetch_pbp(gid: str) -> Dict[str, Any]:
    """Fetch play-by-play JSON for the given game id."""
    return _fetch_json(PBP_URL.format(gid=gid))


def fetch_box(gid: str) -> Dict[str, Any]:
    """Fetch box score JSON for the given game id."""
    return _fetch_json(BOX_URL.format(gid=gid))


def fetch_shotchart(gid: str) -> Dict[str, Any]:
    """Fetch shot chart JSON for the given game id."""
    return _fetch_json(SHOTCHART_URL.format(gid=gid))


def fetch_schedule() -> Dict[str, Any]:
    """Fetch the league schedule."""
    return _fetch_json(SCH_URL)
