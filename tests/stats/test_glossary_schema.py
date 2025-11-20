from __future__ import annotations

import pandas as pd
import pytest

from nba_scraper.stats import PbP, load_glossary_schema

TYPE_FIELD = "Type"
COLUMN_FIELD = "Column"

TYPE_MAP = {
    "int": ("i", "u"),
    "float": ("f",),
    "string": ("O", "S", "U"),
    "bool": ("b",),
    "number": ("i", "u", "f"),
}


def _build_box_from_df(pbp_df: pd.DataFrame) -> pd.DataFrame:
    pbp = PbP(pbp_df)
    box = pbp.player_box_glossary_with_totals()
    assert not box.empty
    return box


def test_glossary_schema_columns_present(v2_pbp_df, cdn_0021900151_df):
    schema = load_glossary_schema()

    mask = schema[TYPE_FIELD].astype(str).str.strip().str.lower().isin(TYPE_MAP.keys())
    implemented = schema[mask]
    columns = implemented[COLUMN_FIELD].dropna().astype(str).tolist()

    box_v2 = _build_box_from_df(v2_pbp_df)
    box_cdn = _build_box_from_df(cdn_0021900151_df)

    for col in columns:
        assert col in box_v2.columns, f"Schema column {col!r} missing from v2 box output"
        assert col in box_cdn.columns, f"Schema column {col!r} missing from CDN box output"


@pytest.mark.parametrize("fixture_name", ["v2_pbp_df", "cdn_0021900151_df"])
def test_glossary_schema_dtypes_match(request, fixture_name):
    schema = load_glossary_schema()

    pbp_df = request.getfixturevalue(fixture_name)
    box = _build_box_from_df(pbp_df)

    for _, row in schema.iterrows():
        col = str(row[COLUMN_FIELD])
        type_str = str(row[TYPE_FIELD]).strip().lower()

        if type_str not in TYPE_MAP:
            continue

        assert col in box.columns, f"Expected column {col!r} in box output"

        dtype = box[col].dtype
        kind = getattr(dtype, "kind", None)

        if kind is None:
            dtype_str = str(dtype).lower()
            if "int" in dtype_str:
                kind = "i"
            elif "float" in dtype_str:
                kind = "f"
            elif "bool" in dtype_str:
                kind = "b"
            elif "string" in dtype_str or "object" in dtype_str:
                kind = "O"

        acceptable = TYPE_MAP[type_str]
        assert kind in acceptable, f"Column {col!r} has dtype {dtype}, expected category {type_str}"


def test_glossary_schema_is_well_formed():
    schema = load_glossary_schema()

    assert COLUMN_FIELD in schema.columns
    assert TYPE_FIELD in schema.columns

    cols = schema[COLUMN_FIELD].astype(str).tolist()
    assert len(cols) == len(set(cols)), "Duplicate column names in glossary_schema.csv"
