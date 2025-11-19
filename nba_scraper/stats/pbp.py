from datetime import datetime
import math
import numpy as np
import pandas as pd
from .box_glossary import (
    annotate_events,
    accumulate_player_counts,
    compute_on_court_exposures,
    build_player_box,
    append_team_totals,
    _coerce_id_scalar,
)


def normalize_game_id(game_id):
    """
    Accept int or string game IDs of length 8 or 10 and return a canonical int.
    Examples:
        "0022200001" -> 22200001
        "22200001"   -> 22200001
        22200001     -> 22200001
    """
    if game_id is None:
        return None
    return int(str(game_id))


def format_game_id(game_id):
    """
    Format a normalized game_id as a 10-digit string with leading zeros.
    Example:
        22200001 -> "0022200001"
    """
    if game_id is None:
        return None
    return f"{int(game_id):010d}"

# NOTE:
# Earlier versions of this module hand-rolled per-player and team aggregations
# directly from the v2 PbP schema. The modern ingestion layer normalizes all
# events into a Canonical DataFrame and the calculations now live entirely in
# box_glossary. PbP is therefore a thin wrapper over that functionality plus the
# possession utilities.


class PbP:
    """
    This class represents one game of of an NBA play by play dataframe. I am
    building methods on top of this class to streamline the calculation of
    stats from the play by player and then insertion into a database of the
    users choosing
    """

    def __init__(self, pbp_df):
        self.df = pbp_df

        id_columns = [
            "game_id",
            "team_id",
            "home_team_id",
            "away_team_id",
            "player1_team_id",
            "player2_team_id",
            "player3_team_id",
            "player1_id",
            "player2_id",
            "player3_id",
            "assist_id",
            *(f"home_player_{i}_id" for i in range(1, 6)),
            *(f"away_player_{i}_id" for i in range(1, 6)),
        ]

        for col in id_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce").astype(
                    "Int64"
                )

        # Enforce single-game input; many invariants assume one game_id.
        game_ids = self.df["game_id"].unique()
        if len(game_ids) != 1:
            raise ValueError(
                f"PbP expects a single-game DataFrame, but found game_ids={game_ids}"
            )
        # Normalize 8- or 10-character IDs; store as int internally
        normalized_game_id = normalize_game_id(game_ids[0])
        self.game_id = normalized_game_id
        self.df["game_id"] = normalized_game_id
        self.home_team = self.df["home_team_abbrev"].unique()[0]
        self.away_team = self.df["away_team_abbrev"].unique()[0]
        self.home_team_id = int(self.df["home_team_id"].unique()[0])
        self.away_team_id = int(self.df["away_team_id"].unique()[0])
        self.season = self.df["season"].unique()[0]

        # Handle PbP classes created from imported CSV files versus those
        # created by nba_scraper that handles game_date as a proper datetime.
        if self.df["game_date"].dtypes == "O":
            raw = str(pbp_df["game_date"].unique()[0])
            parsed = None
            for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
                try:
                    parsed = datetime.strptime(raw, fmt)
                    break
                except ValueError:
                    continue
            if parsed is None:
                # Fallback: let pandas try to infer.
                parsed = pd.to_datetime(raw)
            self.game_date = parsed
            self.df["game_date"] = pd.to_datetime(self.df["game_date"])
        else:
            self.game_date = pbp_df["game_date"].unique()[0]

        # change column types to fit my database at a later time on insert

        self.df["scoremargin"] = self.df["scoremargin"].astype(str)

        # calculating home and away possesions to later aggregate for players
        # and teams

        # NOTE: The home_possession / away_possession flags below implement the
        # original v2-era heuristics for detecting possession boundaries. New
        # possession-based analysis (RAPM, on/off exposures, etc.) should prefer
        # the _build_possessions() output as the canonical representation of
        # possessions. These columns are kept for backwards compatibility.

        # Prefer canonical possession_after when available (CDN + modern schema),
        # fall back to legacy text-based heuristics otherwise.
        if "possession_after" in self.df.columns and self.df["possession_after"].notna().any():
            # Start with zeros.
            self.df["home_possession"] = 0
            self.df["away_possession"] = 0

            home_id = self.home_team_id
            away_id = self.away_team_id

            pos_raw = self.df["possession_after"]

            # Normalize: treat only home/away IDs as valid possession owners.
            # Forward-fill to cover sequences of events where possession doesn't change.
            pos_team = pos_raw.where(pos_raw.isin([home_id, away_id]))
            pos_team = pos_team.ffill()

            # For any leading NaNs (pre-jump-ball) that are still NaN after ffill,
            # backfill them with the first valid possession owner so those events
            # belong to the first possession.
            pos_team = pos_team.bfill()

            # At this point, pos_team is constant over stretches where the same
            # team has the ball. The end of each stretch is a possession boundary.
            is_last_in_stretch = (pos_team != pos_team.shift(-1)) | pos_team.isna()

            # Mark the final event of each possession for home vs away.
            home_pos_idx = is_last_in_stretch & (pos_team == home_id)
            away_pos_idx = is_last_in_stretch & (pos_team == away_id)

            self.df.loc[home_pos_idx, "home_possession"] = 1
            self.df.loc[away_pos_idx, "away_possession"] = 1

        else:
            # --- LEGACY HEURISTIC POSSESSION FLAGS (existing v2-style code path) ---

            # calculating made shot possessions
            self.df["home_possession"] = np.where(
                (self.df.event_team == self.df.home_team_abbrev)
                & (self.df.event_type_de == "shot"),
                1,
                0,
            )

            # calculating turnover possessions
            self.df["home_possession"] = np.where(
                (self.df.event_team == self.df.home_team_abbrev)
                & (self.df.event_type_de == "turnover"),
                1,
                self.df["home_possession"],
            )

            # calculating defensive rebound possessions
            self.df["home_possession"] = np.where(
                (
                    (self.df.event_team == self.df.away_team_abbrev)
                    & (self.df.is_d_rebound == 1)
                )
                | (
                    (self.df.event_type_de == "rebound")
                    & (self.df.is_d_rebound == 0)
                    & (self.df.is_o_rebound == 0)
                    & (self.df.event_team == self.df.away_team_abbrev)
                    & (self.df.event_type_de.shift(1) != "free-throw")
                ),
                1,
                self.df["home_possession"],
            )

            # Determine if it's the last free throw using structured data if available
            is_last_ft = pd.Series(False, index=self.df.index)

            # Ensure descriptions are treated as strings and handle NaNs
            homedesc = self.df.get("homedescription", pd.Series(index=self.df.index)).fillna("").astype(str)
            visitordesc = self.df.get("visitordescription", pd.Series(index=self.df.index)).fillna("").astype(str)

            # Heuristic fallback using string matching
            string_match_last_ft = (
                homedesc.str.contains("Free Throw 1 of 1")
                | homedesc.str.contains("Free Throw 2 of 2")
                | homedesc.str.contains("Free Throw 3 of 3")
                | visitordesc.str.contains("Free Throw 1 of 1")
                | visitordesc.str.contains("Free Throw 2 of 2")
                | visitordesc.str.contains("Free Throw 3 of 3")
            )

            if "ft_n" in self.df.columns and "ft_m" in self.df.columns:
                try:
                    ft_n = self.df["ft_n"].fillna(0).astype(int)
                    ft_m = self.df["ft_m"].fillna(0).astype(int)
                    structured_last_ft = (ft_n == ft_m) & (ft_n > 0)
                    # Use structured data if valid, otherwise fallback to string match
                    is_last_ft = np.where(
                        (ft_n > 0) & (ft_m > 0),
                        structured_last_ft,
                        string_match_last_ft,
                    )
                except (ValueError, TypeError):
                    # Fallback if structured data is malformed
                    is_last_ft = string_match_last_ft
            else:
                # Fallback if columns are missing
                is_last_ft = string_match_last_ft

            # calculating final free throw possessions (Home)
            self.df["home_possession"] = np.where(
                (self.df.event_team == self.df.home_team_abbrev)
                & (self.df.event_type_de == "free-throw")
                & is_last_ft,
                1,
                self.df["home_possession"],
            )

            # calculating made shot possessions (Away)
            self.df["away_possession"] = np.where(
                (self.df.event_team == self.df.away_team_abbrev)
                & (self.df.event_type_de == "shot"),
                1,
                0,
            )

            # calculating turnover possessions (Away)
            self.df["away_possession"] = np.where(
                (self.df.event_team == self.df.away_team_abbrev)
                & (self.df.event_type_de == "turnover"),
                1,
                self.df["away_possession"],
            )

            # calculating defensive rebound possessions (Away)
            self.df["away_possession"] = np.where(
                (
                    (self.df.event_team == self.df.home_team_abbrev)
                    & (self.df.is_d_rebound == 1)
                )
                | (
                    (self.df.event_type_de == "rebound")
                    & (self.df.is_d_rebound == 0)
                    & (self.df.is_o_rebound == 0)
                    & (self.df.event_team == self.df.home_team_abbrev)
                    & (self.df.event_type_de.shift(1) != "free-throw")
                ),
                1,
                self.df["away_possession"],
            )

            # calculating final free throw possessions (Away)
            self.df["away_possession"] = np.where(
                (self.df.event_team == self.df.away_team_abbrev)
                & (self.df.event_type_de == "free-throw")
                & is_last_ft,
                1,
                self.df["away_possession"],
            )





















    @staticmethod
    def _get_off_def_teams(last_event: pd.Series) -> tuple[str, str]:
        """
        Determine offensive and defensive team abbreviations for a possession
        based on the last event in that possession.

        Returns (off_abbrev, def_abbrev), where each is a team abbreviation
        (home_team_abbrev or away_team_abbrev). In ambiguous cases, falls back
        to treating the home team as the offense.
        """
        home_abbrev = last_event.get("home_team_abbrev")
        away_abbrev = last_event.get("away_team_abbrev")
        ev_team = last_event.get("event_team")

        # Prefer annotated 'family' if available, otherwise use raw event_type_de.
        ev_type = last_event.get("family") or last_event.get("event_type_de")
        if isinstance(ev_type, str):
            ev_type = ev_type.replace("-", "_").lower()
        else:
            ev_type = ""

        # Default offense/defense mapping mirrors existing _build_possessions logic:
        # - shot / free_throw / turnover: offense is event_team
        # - rebound: depends on OREB/DREB flags
        # - fallback: treat event_team as offense
        if ev_type in ("shot", "miss_shot", "missed_shot", "free_throw", "turnover"):
            off_abbrev = ev_team
        elif ev_type == "rebound":
            # Check rebound type flags (Critical Fix)
            is_d_rebound = last_event.get("is_d_rebound") == 1
            is_o_rebound = last_event.get("is_o_rebound") == 1

            if is_o_rebound:
                # Offensive rebound: the rebounding team is offense.
                off_abbrev = ev_team
            elif is_d_rebound:
                # Defensive rebound: the rebounding team is defense, the other team was offense.
                if ev_team == home_abbrev:
                    off_abbrev = away_abbrev
                elif ev_team == away_abbrev:
                    off_abbrev = home_abbrev
                else:
                    # If we can't match, fall back to event team
                    off_abbrev = ev_team
            else:
                # Ambiguous/Team rebound: Treat similar to DREB for determining who *had* the ball.
                if ev_team == home_abbrev:
                    off_abbrev = away_abbrev
                elif ev_team == away_abbrev:
                    off_abbrev = home_abbrev
                else:
                    off_abbrev = ev_team
        else:
            off_abbrev = ev_team

        # Derive defense as the other team, with fallbacks.
        if off_abbrev == home_abbrev:
            def_abbrev = away_abbrev
        elif off_abbrev == away_abbrev:
            def_abbrev = home_abbrev
        else:
            # Fallback: if off_abbrev doesn't match either home/away, try event team.
            if ev_team == home_abbrev:
                off_abbrev = home_abbrev
                def_abbrev = away_abbrev
            elif ev_team == away_abbrev:
                off_abbrev = away_abbrev
                def_abbrev = home_abbrev
            else:
                # Final fallback: default to home offense
                off_abbrev = home_abbrev
                def_abbrev = away_abbrev

        return off_abbrev, def_abbrev

    @staticmethod
    def parse_possessions(poss_list: list[pd.DataFrame]) -> tuple[list[pd.DataFrame], list[int]]:
        """
        Parse each possession segment and create one row for offense team
        and defense team lineups.

        Inputs:
            poss_list   - list of dataframes each one representing one possession

        Outputs:
            parsed_list  - list of single-row dataframes, one per possession
            used_indices - list of integer indices into poss_list corresponding to parsed_list
        """
        parsed_list: list[pd.DataFrame] = []
        used_indices: list[int] = []

        # Define metadata columns to carry through
        metadata_cols = [
            "points_made",
            "home_team_abbrev",
            "event_team",
            "away_team_abbrev",
            "home_team_id",
            "away_team_id",
            "game_id",
            "game_date",
            "season",
        ]

        # Explicit list of player columns so we do not rely on column order
        player_cols = [
            "home_player_1", "home_player_1_id",
            "home_player_2", "home_player_2_id",
            "home_player_3", "home_player_3_id",
            "home_player_4", "home_player_4_id",
            "home_player_5", "home_player_5_id",
            "away_player_1", "away_player_1_id",
            "away_player_2", "away_player_2_id",
            "away_player_3", "away_player_3_id",
            "away_player_4", "away_player_4_id",
            "away_player_5", "away_player_5_id",
        ]

        # Dynamically generate column mappings for renaming
        home_to_off = {c: c.replace("home_", "off_") for c in player_cols if "home_" in c}
        away_to_def = {c: c.replace("away_", "def_") for c in player_cols if "away_" in c}
        home_to_def = {c: c.replace("home_", "def_") for c in player_cols if "home_" in c}
        away_to_off = {c: c.replace("away_", "off_") for c in player_cols if "away_" in c}

        # Combine player and metadata columns for selection
        selected_cols = player_cols + metadata_cols

        # Canonical possession-ending families (after normalization)
        valid_end_families = {
            "rebound",
            "turnover",
            "shot",
            "missed_shot",
            "miss_shot",
            "free_throw",
        }

        for seg_idx, df in enumerate(poss_list):
            if df.empty:
                continue

            # Normalize event "family" for robust CDN/v2 handling.
            if "family" in df.columns:
                fam = df["family"].fillna("").astype(str).str.lower()
            else:
                et = df.get("event_type_de")
                if et is None:
                    fam = pd.Series([""] * len(df), index=df.index)
                else:
                    fam = (
                        et.fillna("")
                        .astype(str)
                        .str.lower()
                        .str.replace("-", "_", regex=False)
                    )

            last_fam = fam.iloc[-1]
            if last_fam not in valid_end_families:
                end_idx = fam[fam.isin(valid_end_families)].index
                if len(end_idx) == 0:
                    # No obvious possession-ending event in this segment; skip it for
                    # possession-level RAPM/on‑court scoring while leaving the events
                    # available for timing-based exposures.
                    continue
                df = df.loc[: end_idx[-1]].copy()
                fam = fam.loc[df.index]

            # Fail fast with a clear error if the input doesn't have the
            # expected player columns (e.g., incompatible nba_scraper version).
            missing = [c for c in player_cols if c not in df.columns]
            if missing:
                raise KeyError(
                    "parse_possessions expected player columns "
                    f"{player_cols}, but these are missing: {missing}"
                )

            last_event = df.iloc[-1]

            # Determine offense/defense using centralized helper
            off_abbrev, _ = PbP._get_off_def_teams(last_event)

            def append_possession(off_abbrev_inner: str) -> None:
                home_abbrev = last_event.get("home_team_abbrev")
                away_abbrev = last_event.get("away_team_abbrev")

                if off_abbrev_inner not in (home_abbrev, away_abbrev):
                    # Fallback: assume home is on offense to preserve row count
                    off_abbrev_inner = home_abbrev

                # Create a DataFrame from the last event's relevant columns
                # Handle potential missing columns gracefully
                data = {col: last_event.get(col) for col in selected_cols}
                row_df = pd.DataFrame([data])

                # Rename 'event_team' to 'event_team_abbrev' for consistency if needed
                if "event_team" in row_df.columns and "event_team_abbrev" not in row_df.columns:
                    row_df = row_df.rename(columns={"event_team": "event_team_abbrev"})

                if off_abbrev_inner == home_abbrev:
                    # Home team on offense
                    row_df = row_df.rename(columns={**home_to_off, **away_to_def})
                else:
                    # Away team on offense
                    row_df = row_df.rename(columns={**away_to_off, **home_to_def})

                parsed_list.append(row_df)

            append_possession(off_abbrev)

            # Track which segment produced this possession row
            used_indices.append(seg_idx)

        return parsed_list, used_indices

    def _build_possessions(self, df: pd.DataFrame, include_event_agg: bool = False):
        """
        Internal helper used by rapm_possessions() and compute_on_court_exposures().

        Returns a DataFrame with one row per possession. When include_event_agg is
        True, possession-level shooting aggregates for the offense and defense are
        added.
        """

        # Annotate events once for the whole game
        pbp_df = annotate_events(df.copy())

        # Ensure 0..N index so that label-based indexing from poss_index matches positional iloc
        pbp_df = pbp_df.reset_index(drop=True)

        poss_index = pbp_df[(pbp_df.home_possession == 1) | (pbp_df.away_possession == 1)].index
        shift_dfs = []
        past_index = -1

        for i in poss_index:
            # Slice events between possession markers, skipping empty segments
            seg = pbp_df.iloc[past_index + 1 : i + 1, :].reset_index(drop=True)
            if not seg.empty:
                shift_dfs.append(seg)
            past_index = i

        # Capture any remaining events after the last possession marker
        if past_index < len(pbp_df) - 1:
            seg = pbp_df.iloc[past_index + 1 :].reset_index(drop=True)
            if not seg.empty:
                shift_dfs.append(seg)

        # --- Normalize segments so parse_possessions and event_aggs see the same slice ---
        # Use the canonical 'family' column from annotate_events so we don't depend
        # on v2-style event_type_de strings.
        valid_end_families = {
            "rebound",
            "turnover",
            "shot",
            "missed_shot",
            "miss_shot",
            "free_throw",
        }

        normalized_segments: list[pd.DataFrame] = []
        for seg in shift_dfs:
            if seg.empty:
                normalized_segments.append(seg)
                continue

            # Prefer the canonical family column when available; otherwise fall back
            # to a normalized event_type_de.
            if "family" in seg.columns:
                fam = seg["family"].fillna("").astype(str).str.lower()
            else:
                et = seg.get("event_type_de")
                if et is None:
                    fam = pd.Series([""] * len(seg), index=seg.index)
                else:
                    fam = (
                        et.fillna("")
                        .astype(str)
                        .str.lower()
                        .str.replace("-", "_", regex=False)
                    )

            last_fam = fam.iloc[-1]
            if last_fam in valid_end_families:
                normalized_segments.append(seg)
                continue

            mask = fam.isin(valid_end_families)
            if not mask.any():
                # No obvious possession-ending event in this stretch:
                # treat as non-possession for RAPM/on‑court scoring.
                normalized_segments.append(seg.iloc[0:0])
                continue

            last_idx = mask[mask].index[-1]
            normalized_segments.append(seg.loc[: last_idx].copy())

        # Use normalized segments everywhere downstream
        parsed_possessions, used_indices = self.parse_possessions(normalized_segments)

        # If no possessions were detected, return an empty DataFrame
        if not parsed_possessions:
            return pd.DataFrame()

        event_aggs = []
        for poss_df in normalized_segments:
            poss_events = poss_df  # already annotated via pbp_df
            if poss_events.empty:
                event_aggs.append(
                    {
                        "off_team_id": np.nan,
                        "off_team_abbrev": "",
                        "def_team_id": np.nan,
                        "def_team_abbrev": "",
                        "off_team_FGA": 0,
                        "off_team_FGM": 0,
                        "off_team_3PA": 0,
                        "off_team_3PM": 0,
                        "off_team_2PA": 0,
                        "off_team_FTA": 0,
                        "off_team_FTM": 0,
                        "def_team_FGA": 0,
                        "def_team_FGM": 0,
                        "def_team_3PA": 0,
                        "def_team_3PM": 0,
                        "def_team_2PA": 0,
                        "def_team_FTA": 0,
                        "def_team_FTM": 0,
                        "points_for_offense": 0,
                        "points_for_defense": 0,
                    }
                )
                continue

            last_event = poss_events.iloc[-1]
            
            home_abbrev = last_event.get("home_team_abbrev")
            away_abbrev = last_event.get("away_team_abbrev")

            # Centralized determination
            off_abbrev, def_abbrev = self._get_off_def_teams(last_event)

            off_team_id = (
                last_event.get("home_team_id")
                if off_abbrev == home_abbrev
                else last_event.get("away_team_id")
            )
            def_team_id = (
                last_event.get("away_team_id")
                if off_abbrev == home_abbrev
                else last_event.get("home_team_id")
            )

            off_mask = poss_events.get("team_id") == off_team_id
            def_mask = poss_events.get("team_id") == def_team_id

            off_fga = poss_events.loc[off_mask, "is_fg_attempt"].sum()
            off_fgm = poss_events.loc[off_mask, "is_fg_make"].sum()
            off_3pa_mask = (
                poss_events.loc[off_mask, "is_fg_attempt"].astype(bool)
                & poss_events.loc[off_mask, "is_three"].astype(bool)
            )
            off_3pm_mask = (
                poss_events.loc[off_mask, "is_fg_make"].astype(bool)
                & poss_events.loc[off_mask, "is_three"].astype(bool)
            )
            off_2pa_mask = (
                poss_events.loc[off_mask, "is_fg_attempt"].astype(bool)
                & ~poss_events.loc[off_mask, "is_three"].astype(bool)
            )

            def_fga = poss_events.loc[def_mask, "is_fg_attempt"].sum()
            def_fgm = poss_events.loc[def_mask, "is_fg_make"].sum()
            def_3pa_mask = (
                poss_events.loc[def_mask, "is_fg_attempt"].astype(bool)
                & poss_events.loc[def_mask, "is_three"].astype(bool)
            )
            def_3pm_mask = (
                poss_events.loc[def_mask, "is_fg_make"].astype(bool)
                & poss_events.loc[def_mask, "is_three"].astype(bool)
            )
            def_2pa_mask = (
                poss_events.loc[def_mask, "is_fg_attempt"].astype(bool)
                & ~poss_events.loc[def_mask, "is_three"].astype(bool)
            )

            off_points = poss_events.loc[off_mask, "points_made"].sum()
            def_points = poss_events.loc[def_mask, "points_made"].sum()

            event_aggs.append(
                {
                    "off_team_id": off_team_id,
                    "off_team_abbrev": off_abbrev,
                    "def_team_id": def_team_id,
                    "def_team_abbrev": def_abbrev,
                    "off_team_FGA": off_fga,
                    "off_team_FGM": off_fgm,
                    "off_team_3PA": off_3pa_mask.sum(),
                    "off_team_3PM": off_3pm_mask.sum(),
                    "off_team_2PA": off_2pa_mask.sum(),
                    "off_team_FTA": poss_events.loc[off_mask, "is_ft"].sum(),
                    "off_team_FTM": poss_events.loc[off_mask, "is_ft_make"].sum(),
                    "def_team_FGA": def_fga,
                    "def_team_FGM": def_fgm,
                    "def_team_3PA": def_3pa_mask.sum(),
                    "def_team_3PM": def_3pm_mask.sum(),
                    "def_team_2PA": def_2pa_mask.sum(),
                    "def_team_FTA": poss_events.loc[def_mask, "is_ft"].sum(),
                    "def_team_FTM": poss_events.loc[def_mask, "is_ft_make"].sum(),
                    "points_for_offense": off_points,
                    "points_for_defense": def_points,
                }
            )

        poss_df = pd.concat(parsed_possessions, sort=True).reset_index(drop=True)

        if event_aggs:
            agg_df = pd.DataFrame(event_aggs)

            # Robustly align aggregates to the parsed possessions using the returned indices
            agg_df = agg_df.iloc[used_indices].reset_index(drop=True)

            # Attach team IDs and abbreviations inferred in the event_aggs loop
            poss_df["off_team_id"] = agg_df["off_team_id"].values
            poss_df["def_team_id"] = agg_df["def_team_id"].values
            poss_df["off_team_abbrev"] = agg_df["off_team_abbrev"].values
            poss_df["def_team_abbrev"] = agg_df["def_team_abbrev"].values

            poss_df["points_for_offense"] = agg_df["points_for_offense"].values
            poss_df["points_for_defense"] = agg_df["points_for_defense"].values

            if include_event_agg:
                for col in [
                    "off_team_FGA",
                    "off_team_FGM",
                    "off_team_3PA",
                    "off_team_3PM",
                    "off_team_2PA",
                    "off_team_FTA",
                    "off_team_FTM",
                    "def_team_FGA",
                    "def_team_FGM",
                    "def_team_3PA",
                    "def_team_3PM",
                    "def_team_2PA",
                    "def_team_FTA",
                    "def_team_FTM",
                ]:
                    poss_df[col] = agg_df[col].values
        else:
            # Fallback when no events were parsed
            poss_df["off_team_id"] = np.nan
            poss_df["def_team_id"] = np.nan
            poss_df["off_team_abbrev"] = ""
            poss_df["def_team_abbrev"] = ""
            poss_df["points_for_offense"] = 0
            poss_df["points_for_defense"] = 0

        # Backwards-compatibility: expose scoring in a single points column too.
        poss_df["points_made"] = poss_df["points_for_offense"]

        return poss_df

    def rapm_possessions(self):
        """
        Extract out all the RAPM possessions as a DataFrame.

        This uses the same event-level scoring logic as the on-court glossary:
        - points_for_offense: total offensive points scored in the possession.
        - points_for_defense: total points scored by the opponent in the possession.

        For backward compatibility, a 'points_made' column is also provided and
        is set equal to points_for_offense by _build_possessions.
        """
        pbp_df = self.df.copy()
        poss_df = self._build_possessions(pbp_df, include_event_agg=True)
        return poss_df

    def _compute_starters(self) -> pd.DataFrame:
        """
        Return a DataFrame with one row per (game_id, team_id, player_id)
        marking whether the player started the game (Starts = 1).

        Assumes this PbP instance is a single game.
        """
        df = self.df.copy()

        # Sort by game clock so we consistently pick the earliest event
        sort_cols = [c for c in ["period", "seconds_elapsed", "eventnum"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols)

        lineup_cols = [
            *(f"home_player_{i}_id" for i in range(1, 6)),
            *(f"away_player_{i}_id" for i in range(1, 6)),
        ]

        mask = pd.Series(True, index=df.index)
        for col in lineup_cols:
            if col in df.columns:
                mask &= df[col].notna() & (df[col] != 0)

        lineup_df = df[mask] if not df.empty else df
        first_row = lineup_df.iloc[0] if not lineup_df.empty else df.iloc[0]

        home_ids = [first_row.get(f"home_player_{i}_id") for i in range(1, 6)]
        away_ids = [first_row.get(f"away_player_{i}_id") for i in range(1, 6)]

        starters = []
        game_id = _coerce_id_scalar(first_row.get("game_id"))
        home_team_id = _coerce_id_scalar(first_row.get("home_team_id"))
        away_team_id = _coerce_id_scalar(first_row.get("away_team_id"))

        for pid in home_ids:
            if pid and pid != 0 and not pd.isna(pid):
                starters.append(
                    {
                        "game_id": game_id,
                        "team_id": home_team_id,
                        "player_id": _coerce_id_scalar(pid),
                        "Starts": 1,
                    }
                )
        for pid in away_ids:
            if pid and pid != 0 and not pd.isna(pid):
                starters.append(
                    {
                        "game_id": game_id,
                        "team_id": away_team_id,
                        "player_id": _coerce_id_scalar(pid),
                        "Starts": 1,
                    }
                )

        starters_df = pd.DataFrame(starters)
        if not starters_df.empty:
            for col in ["game_id", "team_id", "player_id"]:
                if col in starters_df.columns:
                    starters_df[col] = pd.to_numeric(
                        starters_df[col], errors="coerce"
                    ).astype("Int64")

        return starters_df

    def player_box_glossary(
        self,
        player_meta: pd.DataFrame | None = None,
        game_meta: pd.DataFrame | None = None,
        player_game_meta: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Build a per-player, per-game box aligned to an external glossary.

        Assumes this PbP instance represents a single game.

        - Inputs:
            player_meta: optional DataFrame with identity/biographical fields
                keyed by NBA.com personId (player_id / NbaDotComID).
            game_meta: optional DataFrame with game-level fields keyed by game_id.

        - Output:
            DataFrame with one row per (game_id, team_id, player_id),
            including raw counts and derived rates (TS, USG, ORB%, DRB%, AST%, BLK%, etc.)
            plus on/off stats.
        """
        df = self.df.copy()

        df = annotate_events(df)
        counts_df = accumulate_player_counts(df)
        exposures_df = compute_on_court_exposures(self, df)

        # Defensively filter player_meta to the current season to avoid duplicate rows
        pm = player_meta
        if pm is not None and not pm.empty:
            pm = pm.copy()
            # If player_meta includes a 'season' or 'Year' column, trim to this game's season
            if "season" in pm.columns:
                pm = pm[pm["season"] == self.season]
            elif "Year" in pm.columns:
                pm = pm[pm["Year"] == self.season]

            # Ensure unique row per player identifier
            if "NbaDotComID" in pm.columns:
                pm = pm.drop_duplicates(subset=["NbaDotComID"])
            elif "player_id" in pm.columns:
                pm = pm.drop_duplicates(subset=["player_id"])

        box_df = build_player_box(
            df=df,
            counts_df=counts_df,
            exposures_df=exposures_df,
            player_meta=pm,
            game_meta=game_meta,
            player_game_meta=player_game_meta,
        )

        # Normalize potential merge artifacts from player_meta
        if "player_id_x" in box_df.columns:
            box_df["player_id"] = box_df["player_id_x"]
            box_df.drop(columns=[c for c in ["player_id_x", "player_id_y"] if c in box_df.columns], inplace=True)

        # Fill Starts from starting lineups
        starters_df = self._compute_starters()
        if not starters_df.empty:
            # Preserve original Starts column if it exists, then merge
            if "Starts" in box_df.columns:
                box_df.rename(columns={"Starts": "Starts_x"}, inplace=True)
            box_df = box_df.merge(
                starters_df, on=["game_id", "team_id", "player_id"], how="left"
            )
            # Coalesce merged 'Starts' with original, fill NaNs with 0
            box_df["Starts"] = box_df["Starts"].fillna(box_df.get("Starts_x", 0)).fillna(0).astype(int)
            box_df.drop(columns=[c for c in box_df.columns if c.endswith("_x") or c == 'Starts_y'], inplace=True)

        # Sanity check: on-court points must match team totals.
        self._check_on_court_points_consistency(box_df)
        self._check_on_court_minutes_consistency(box_df)

        return box_df

    def player_box_glossary_with_totals(
        self,
        player_meta: pd.DataFrame | None = None,
        game_meta: pd.DataFrame | None = None,
        player_game_meta: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Convenience wrapper: build the per-player glossary box and append one
        'TOTAL' row per (game_id, team_id).
        """
        box = self.player_box_glossary(
            player_meta=player_meta,
            game_meta=game_meta,
            player_game_meta=player_game_meta,
        )
        return append_team_totals(box)

    def _check_on_court_points_consistency(self, box: pd.DataFrame, tol: float = 1e-6) -> None:
        """
        Internal helper: verify that summed OnCourt_Team_Points / OnCourt_Opp_Points
        per team match the scoreboard totals times 5 for this game.

        Raises AssertionError if an invariant is violated.
        """
        # Build a simple mapping team_id -> points_for for this game using the
        # canonical points_made column. This works for full-game and partial
        # play-by-play feeds alike.
        if "points_made" not in self.df.columns or "team_id" not in self.df.columns:
            return

        points = pd.to_numeric(self.df["points_made"], errors="coerce").fillna(0)
        teams = self.df["team_id"]
        points_df = pd.DataFrame({"team_id": teams, "points": points})
        points_df = points_df[(points_df["team_id"].notna()) & (points_df["team_id"] != 0)]
        team_points = points_df.groupby("team_id")["points"].sum()

        if team_points.empty:
            return

        team_points_map = {int(t): float(val) for t, val in team_points.items()}

        if not team_points_map:
            return

        for team_id, points_for in team_points_map.items():
            expected_for = points_for * 5.0
            actual_for = box.loc[box["team_id"] == team_id, "OnCourt_Team_Points"].sum()

            if abs(actual_for - expected_for) > tol:
                raise AssertionError(
                    f"OnCourt_Team_Points inconsistency for team {team_id}: "
                    f"actual={actual_for}, expected={expected_for}"
                )

            # Opponent points are the sum of all other teams' points_for.
            opponent_points = sum(
                p for t, p in team_points_map.items() if t != team_id
            )
            expected_against = opponent_points * 5.0
            actual_against = box.loc[box["team_id"] == team_id, "OnCourt_Opp_Points"].sum()

            if abs(actual_against - expected_against) > tol:
                raise AssertionError(
                    f"OnCourt_Opp_Points inconsistency for team {team_id}: "
                    f"actual={actual_against}, expected={expected_against}"
                )

    def _check_on_court_minutes_consistency(self, box: pd.DataFrame, tol: float = 1e-6) -> None:
        """
        Internal helper: verify that summed Minutes per team equal total game
        duration multiplied by 5.

        Raises AssertionError if an invariant is violated.
        """

        if "event_length" not in self.df.columns or self.df.empty:
            return

        durations = self.df.copy()
        durations["event_length"] = durations["event_length"].fillna(0).astype(float)
        game_minutes = durations.groupby("game_id")["event_length"].sum() / 60.0

        for game_id, total_minutes in game_minutes.items():
            expected_minutes = total_minutes * 5.0
            team_minutes = box.loc[box["game_id"] == game_id].groupby("team_id")["Minutes"].sum()

            for team_id, actual_minutes in team_minutes.items():
                if abs(actual_minutes - expected_minutes) > tol:
                    raise AssertionError(
                        f"OnCourt Minutes inconsistency for team {team_id} in game {game_id}: "
                        f"actual={actual_minutes}, expected={expected_minutes}"
                    )


