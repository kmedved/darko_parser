import pandas as pd


def backfill_coords_with_shotchart(
    df: pd.DataFrame, shotchart_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Backfill x/y/shot_distance for shots missing coordinates using a shot chart
    dataframe containing ['game_id','eventnum','x','y','shot_distance'].
    """
    if df.empty or shotchart_df.empty:
        return df

    key = ["game_id", "eventnum"]
    right = (
        shotchart_df[key + ["x", "y", "shot_distance"]]
        .dropna(subset=["x", "y"], how="any")
        .drop_duplicates(key, keep="last")
    )
    if right.empty:
        return df

    merged = df.merge(right, on=key, how="left", suffixes=("", "_sc"))
    style_flags_col = "style_flags" in merged.columns

    def _has_xy_synth(flags) -> bool:
        if isinstance(flags, (list, tuple, set)):
            return "xy_synth" in flags
        return False

    synth_mask = (
        merged["style_flags"].apply(_has_xy_synth)
        if style_flags_col
        else pd.Series(False, index=merged.index)
    )
    has_sc_xy = merged["x_sc"].notna() & merged["y_sc"].notna()
    needs_xy_override = (merged["x"].isna() | merged["y"].isna() | synth_mask) & has_sc_xy

    for col in ("x", "y", "shot_distance"):
        sc_col = f"{col}_sc"
        if sc_col not in merged.columns:
            continue
        if col in {"x", "y"}:
            take_mask = needs_xy_override
        else:
            take_mask = merged[col].isna() & merged[sc_col].notna()
        merged.loc[take_mask, col] = merged.loc[take_mask, sc_col]
        merged.drop(columns=[sc_col], inplace=True)

    if style_flags_col and needs_xy_override.any():
        def _drop_xy_synth(flags):
            if not flags:
                return flags
            updated = [flag for flag in flags if flag != "xy_synth"]
            return updated

        merged.loc[needs_xy_override, "style_flags"] = merged.loc[
            needs_xy_override, "style_flags"
        ].apply(_drop_xy_synth)

    return merged
