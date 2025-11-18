"""High level scrape helpers that delegate to the unified parsing layer."""
from __future__ import annotations

import os
from typing import Optional

from . import io_sources


def main_scrape(game_id: str, mapping_yaml_path: Optional[str] = None):
    mapping_yaml_path = mapping_yaml_path or os.getenv("NBA_SCRAPER_MAP")
    df = io_sources.parse_any(game_id, io_sources.SourceKind.CDN_REMOTE, mapping_yaml_path)
    return df
