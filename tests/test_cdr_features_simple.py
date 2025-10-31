import pandas as pd
import pytest

from workspace.bioinformatics.cdr_features_simple import (
    H3_CATEGORY_ENCODING,
    calculate_h3_flexibility,
    categorize_h3_length,
    extract_basic_cdr_features,
)


def test_extract_basic_cdr_features_expected_lengths():
    df = pd.DataFrame(
        {
            "antibody_id": ["AB1"],
            "vh_protein_sequence": ["A" * 130],
            "vl_protein_sequence": ["C" * 120],
        }
    )

    result = extract_basic_cdr_features(df)
    row = result.iloc[0]

    assert row["extraction_success"]
    assert row["cdr_h1_length"] == 10
    assert row["cdr_h3_length"] == 13
    assert row["cdr_l1_length"] == 8
    assert row["cdr_l3_length"] == 9
    assert row["h3_category"] == "medium"
    assert row["h3_category_code"] == H3_CATEGORY_ENCODING["medium"]
    assert row["cdr_total_net_charge"] == pytest.approx(0.0)
    assert row["cdr_total_glycosylation_sites"] == 0


def test_calculate_h3_flexibility_scoring():
    score = calculate_h3_flexibility("GSTP")
    assert score == pytest.approx(0.5)


@pytest.mark.parametrize(
    "length, expected",
    [
        (6, "short"),
        (10, "medium"),
        (18, "long"),
        (25, "ultralong"),
    ],
)
def test_categorize_h3_length(length, expected):
    assert categorize_h3_length(length) == expected
