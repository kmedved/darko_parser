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
    """
    def __init__(self, df: pd.DataFrame):
        if df.empty:
            raise ValueError("PbP initialized with empty DataFrame")
        
        # Trust the scraper's schema implicitly
        self.df = df.copy()

        # Validate Single Game Integrity
        game_ids = self.df["game_id"].unique()
        if len(game_ids) != 1:
            raise ValueError(f"PbP expects single-game DF, found: {game_ids}")
        
        self.game_id = int(game_ids[0])
        self.season = int(self.df["season"].iloc[0])
        self.home_team_id = int(self.df["home_team_id"].iloc[0])
        self.away_team_id = int(self.df["away_team_id"].iloc[0])

        self._flag_possessions()

    def _flag_possessions(self):
        """Derived from scraper's possession_after column."""
        self.df["home_possession"] = 0
        self.df["away_possession"] = 0
        
        # possession_after is calculated by nba_scraper
        pos_team = self.df["possession_after"]
        
        # A possession ends when the possession holder changes
        is_change = (pos_team != pos_team.shift(-1))
        
        self.df.loc[is_change & (pos_team == self.home_team_id), "home_possession"] = 1
        self.df.loc[is_change & (pos_team == self.away_team_id), "away_possession"] = 1

    def player_box_glossary(self, player_meta=None, game_meta=None):
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
        # Annotate to get 'family', 'is_d_rebound' normalized
        pbp_df = annotate_events(self.df.copy())
        pbp_df = pbp_df.reset_index(drop=True)

        # 1. Slice DataFrame into chunks based on possession flags
        mask = (pbp_df.home_possession == 1) | (pbp_df.away_possession == 1)
        poss_indices = pbp_df[mask].index
        
        results = []
        past_index = -1
        
        valid_end_families = {"rebound", "turnover", "shot", "missed_shot", "free_throw"}

        # 2. Iterate through possessions
        for i in poss_indices:
            seg = pbp_df.iloc[past_index + 1 : i + 1]
            past_index = i
            
            if seg.empty:
                continue

            # 3. Validate Ending
            last_fam = seg["family"].iloc[-1]
            if last_fam not in valid_end_families:
                continue

            # 4. Determine Offense/Defense
            last_event = seg.iloc[-1]
            ev_team_id = last_event["team_id"]
            
            # Logic: The team performing the action is usually offense,
            # UNLESS it is a defensive rebound.
            off_team_id = ev_team_id
            if last_fam == "rebound" and last_event.get("is_d_rebound"):
                # If Def Rebound, the OTHER team was the one on offense
                if ev_team_id == self.home_team_id:
                    off_team_id = self.away_team_id
                else:
                    off_team_id = self.home_team_id
            
            def_team_id = self.away_team_id if off_team_id == self.home_team_id else self.home_team_id
            
            # 5. Calculate Points for this possession
            # Sum points scored by the OFFENSE team
            points = seg[seg["team_id"] == off_team_id]["points_made"].sum()
            
            row = {
                "game_id": self.game_id,
                "off_team_id": off_team_id,
                "def_team_id": def_team_id,
                "points": points,
                "possessions": 1
            }
            
            # 6. Add Lineups (Player IDs)
            # Note: We take the lineup from the LAST event of the possession
            prefix = "home_" if off_team_id == self.home_team_id else "away_"
            opp_prefix = "away_" if prefix == "home_" else "home_"
            
            for k in range(1, 6):
                row[f"off_player_{k}_id"] = last_event.get(f"{prefix}player_{k}_id", 0)
                row[f"def_player_{k}_id"] = last_event.get(f"{opp_prefix}player_{k}_id", 0)
                
            results.append(row)
            
        return pd.DataFrame(results) if results else pd.DataFrame()