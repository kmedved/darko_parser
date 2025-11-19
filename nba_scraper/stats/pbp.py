import pandas as pd
import numpy as np
from .box_glossary import (
    annotate_events,
    accumulate_player_counts,
    compute_on_court_exposures,
    build_player_box,
)

class PbP:
    """
    Analysis wrapper for a canonical NBA Scraper DataFrame.
    Assumes input df adheres to nba_scraper.schema.CANONICAL_COLUMNS.
    """
    def __init__(self, df: pd.DataFrame):
        if df.empty:
            raise ValueError("PbP initialized with empty DataFrame")
        
        # 1. Trust the Scraper
        # The scraper already handles ID coercion, date parsing, and numeric types.
        self.df = df.copy()

        # 2. Validate Single Game Integrity
        game_ids = self.df["game_id"].unique()
        if len(game_ids) != 1:
            raise ValueError(f"PbP expects single-game DF, found: {game_ids}")
        
        self.game_id = int(game_ids[0])
        self.season = int(self.df["season"].iloc[0])
        
        # 3. Essential Metadata (Guaranteed by scraper schema)
        self.home_team_id = int(self.df["home_team_id"].iloc[0])
        self.away_team_id = int(self.df["away_team_id"].iloc[0])

        # 4. Flag Possessions
        self._flag_possessions()

    def _flag_possessions(self):
        """Derived from scraper's possession_after column."""
        self.df["home_possession"] = 0
        self.df["away_possession"] = 0
        
        # Use the column the scraper explicitly computed for us
        pos_team = self.df["possession_after"]
        
        # Detect switches or end of file
        is_change = (pos_team != pos_team.shift(-1))
        
        # Flag the *last* event of a possession
        self.df.loc[is_change & (pos_team == self.home_team_id), "home_possession"] = 1
        self.df.loc[is_change & (pos_team == self.away_team_id), "away_possession"] = 1

    def player_box_glossary(self, player_meta=None, game_meta=None):
        """
        Main Entry Point: Returns the advanced box score.
        """
        annotated_df = annotate_events(self.df)
        counts = accumulate_player_counts(annotated_df)
        exposures = compute_on_court_exposures(self, annotated_df)
        
        return build_player_box(
            self.df, counts, exposures, 
            player_meta=player_meta, game_meta=game_meta
        )
    
    def rapm_possessions(self):
        """
        Returns RAPM-ready possession rows.
        """
        pbp_df = annotate_events(self.df.copy())
        pbp_df = pbp_df.reset_index(drop=True)

        # Slice DataFrame into chunks based on possession flags
        poss_index = pbp_df[(pbp_df.home_possession == 1) | (pbp_df.away_possession == 1)].index
        
        results = []
        past_index = -1
        
        # Pre-calculate valid endings to speed up the loop
        valid_end_families = {"rebound", "turnover", "shot", "missed_shot", "free_throw"}

        for i in poss_index:
            seg = pbp_df.iloc[past_index + 1 : i + 1]
            if not seg.empty:
                row = self._parse_single_possession(seg, valid_end_families)
                if row:
                    results.append(row)
            past_index = i
            
        return pd.DataFrame(results) if results else pd.DataFrame()

    def _parse_single_possession(self, seg, valid_end_families):
        """
        Aggregates a slice of events into a single RAPM row.
        """
        last_fam = seg["family"].iloc[-1]
        if last_fam not in valid_end_families:
            return None

        last_event = seg.iloc[-1]
        ev_team_id = last_event["team_id"]
        
        # Determine Offense
        off_team_id = ev_team_id
        if last_fam == "rebound":
            if last_event.get("is_d_rebound"):
                # If defensive rebound, the OTHER team was on offense
                off_team_id = self.away_team_id if ev_team_id == self.home_team_id else self.home_team_id
        
        def_team_id = self.away_team_id if off_team_id == self.home_team_id else self.home_team_id
        
        # Calculate Points (Simple Sum)
        points = seg[seg["team_id"] == off_team_id]["points_made"].sum()
        
        row = {
            "game_id": self.game_id,
            "off_team_id": off_team_id,
            "def_team_id": def_team_id,
            "points": points,
            "possessions": 1
        }
        
        # Add lineups
        prefix = "home_" if off_team_id == self.home_team_id else "away_"
        opp_prefix = "away_" if prefix == "home_" else "home_"
        
        for i in range(1, 6):
            row[f"off_player_{i}_id"] = last_event.get(f"{prefix}player_{i}_id", 0)
            row[f"def_player_{i}_id"] = last_event.get(f"{opp_prefix}player_{i}_id", 0)
            
        return row