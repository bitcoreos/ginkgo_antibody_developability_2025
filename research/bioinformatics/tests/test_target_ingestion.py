import pandas as pd

from bioinformatics.target_ingestion import TargetIngestion


def test_load_raw_targets_has_expected_columns():
    ingestion = TargetIngestion()
    targets_df = ingestion.load_raw_targets()

    assert not targets_df.empty
    assert set(ingestion.standard_columns).issubset(targets_df.columns)
    numeric_subset = targets_df[ingestion.RAW_TO_STANDARD.values()]
    assert numeric_subset.dtypes.apply(pd.api.types.is_numeric_dtype).all()
