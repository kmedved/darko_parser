from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nba_scraper.stats import PbP, load_glossary_schema


TYPE_MAP = {
    "int": ("i", "u"),
    "float": ("f", "i", "u"),
    "string": ("O", "S", "U"),
    "bool": ("b",),
}


def _build_box_from_df(pbp_df: pd.DataFrame) -> pd.DataFrame:
    pbp = PbP(pbp_df)
    box = pbp.player_box_glossary_with_totals()
    assert not box.empty
    return box


def test_glossary_schema_columns_present(v2_pbp_df, cdn_0021900151_df):
    schema = load_glossary_schema()
    col_field = "Column"
    columns = schema[col_field].dropna().astype(str).tolist()

    box_v2 = _build_box_from_df(v2_pbp_df)
    box_cdn = _build_box_from_df(cdn_0021900151_df)

    for col in columns:
        assert col in box_v2.columns, f"Schema column {col!r} missing from v2 box output"
        assert col in box_cdn.columns, f"Schema column {col!r} missing from CDN box output"


@pytest.mark.parametrize("fixture_name", ["v2_pbp_df", "cdn_0021900151_df"])
def test_glossary_schema_dtypes_match(request, fixture_name):
    schema = load_glossary_schema()
    col_field = "Column"
    type_field = "Type"

    pbp_df = request.getfixturevalue(fixture_name)
    box = _build_box_from_df(pbp_df)

    for _, row in schema.iterrows():
        col = str(row[col_field])
        type_str = str(row[type_field]).strip().lower()

        if type_str not in TYPE_MAP:
            continue

        assert col in box.columns, f"Expected column {col!r} in box output"

        dtype = box[col].dtype
        kind = getattr(dtype, "kind", None)
        if kind is None and "Int" in str(dtype):
            kind = "i"
        if kind is None and "string" in str(dtype).lower():
            kind = "O"

        acceptable = TYPE_MAP[type_str]
        assert kind in acceptable, f"Column {col!r} has dtype {dtype}, expected {type_str}"


def test_glossary_schema_is_well_formed():
    schema = load_glossary_schema()
    assert "Column" in schema.columns
    assert "Type" in schema.columns

    cols = schema["Column"].astype(str).tolist()
    assert len(cols) == len(set(cols)), "Duplicate column names in glossary_schema.csv"
