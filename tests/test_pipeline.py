import numpy as np
import pandas as pd

from cluster_c.config import CX_UPSELL_GROUP_VALUES, RAW_COLUMNS, XCAT_FLAG_VALUES
from cluster_c.preparation import (
    CATEGORICAL_FEATURES,
    DERIVED_FLOAT32_COLUMNS,
    NUMERIC_FEATURES,
    prepare,
    save_prepared,
)
from cluster_c.synthetic import generate_term_deals


def test_synthetic_data_is_deterministic_and_matches_contract() -> None:
    first = generate_term_deals(seed=42, row_count=200)
    second = generate_term_deals(seed=42, row_count=200)
    pd.testing.assert_frame_equal(first, second)
    assert set(RAW_COLUMNS).issubset(first.columns)
    assert set(first.TERM_BUCKET) == {"1 Year", "3 Year", "5 Year"}
    assert first[["CX_UPSELL_GROUP", "XCAT_FLG", "DV_ATTRIBUTION_CD"]].notna().all().all()
    assert set(first.CX_UPSELL_GROUP).issubset(CX_UPSELL_GROUP_VALUES)
    assert set(first.XCAT_FLG).issubset(XCAT_FLAG_VALUES)
    assert pd.api.types.is_float_dtype(first.DV_ATTRIBUTION_CD)


def test_preparation_passes_and_prior_history_does_not_use_current_target() -> None:
    raw = generate_term_deals(seed=42, row_count=300)
    baseline = prepare(raw).frame
    changed = raw.copy()
    target_id = baseline.loc[
        baseline["CUSTOMER_PRIOR_DEAL_COUNT"] > 0, "SUBSCRIPTION_REFERENCE_ID"
    ].iloc[0]
    changed.loc[changed.SUBSCRIPTION_REFERENCE_ID == target_id, "BOOKINGS_NET_PRICE_USD_AMT"] *= 100
    rebuilt = prepare(changed).frame
    baseline_row = baseline[baseline.SUBSCRIPTION_REFERENCE_ID == target_id].iloc[0]
    rebuilt_row = rebuilt[rebuilt.SUBSCRIPTION_REFERENCE_ID == target_id].iloc[0]
    assert baseline_row.CUSTOMER_PRIOR_AVG_BOOKINGS == rebuilt_row.CUSTOMER_PRIOR_AVG_BOOKINGS
    assert (prepare(raw).checks.status == "PASS").all()


def test_row_level_contract_failures_are_non_blocking() -> None:
    raw = generate_term_deals(seed=42, row_count=30)
    raw.loc[0, ["TERM_BUCKET", "TERM_START_DATE", "BOOKINGS_NET_PRICE_USD_AMT"]] = np.nan

    result = prepare(raw)

    assert len(result.frame) == len(raw)
    assert pd.isna(result.frame.loc[result.frame["SUBSCRIPTION_REFERENCE_ID"] == raw.loc[0, "SUBSCRIPTION_REFERENCE_ID"], "TERM_YEARS"]).all()
    assert "critical_values_present" in set(result.checks.loc[result.checks.status == "FAIL", "check_id"])


def test_common_term_bucket_aliases_are_normalized() -> None:
    raw = generate_term_deals(seed=42, row_count=6)
    raw["TERM_BUCKET"] = ["12 Months", "36_months", "60-MONTHS", "1Y", 3, "5 years"]
    expected = dict(zip(
        raw["SUBSCRIPTION_REFERENCE_ID"],
        ["1 Year", "3 Year", "5 Year", "1 Year", "3 Year", "5 Year"],
    ))

    result = prepare(raw)

    actual = dict(zip(result.frame["SUBSCRIPTION_REFERENCE_ID"], result.frame["TERM_BUCKET"]))
    assert actual == expected


def test_prepared_features_are_saved_as_parquet(tmp_path) -> None:
    result = prepare(generate_term_deals(seed=42, row_count=50))

    paths = save_prepared(result, tmp_path)
    restored = pd.read_parquet(paths["features"], engine="pyarrow")

    assert paths["features"].name == "term_model_features.parquet"
    assert len(restored) == len(result.frame)
    assert restored.columns.tolist() == result.frame.columns.tolist()
    assert pd.api.types.is_datetime64_any_dtype(restored["TERM_START_DATE"])
    assert result.summary["prepared_feature_format"] == "parquet"
    assert result.summary["prepared_feature_compression"] == "snappy"


def test_prepared_data_uses_compact_explicit_dtypes() -> None:
    frame = prepare(generate_term_deals(seed=42, row_count=100)).frame

    assert str(frame["FISCAL_YEAR"].dtype) == "Int16"
    assert str(frame["FISCAL_QUARTER_ID"].dtype) == "Int32"
    assert str(frame["FISCAL_PERIOD_ID"].dtype) == "Int32"
    assert str(frame["TERM_START_MONTH"].dtype) == "Int8"
    assert str(frame["TERM_YEARS"].dtype) == "Int8"
    assert str(frame["CUSTOMER_PRIOR_DEAL_COUNT"].dtype) == "Int32"
    assert all(str(frame[column].dtype) == "float32" for column in DERIVED_FLOAT32_COLUMNS)
    assert all(isinstance(frame[column].dtype, pd.CategoricalDtype) for column in CATEGORICAL_FEATURES)
    assert frame["BOOKINGS_NET_PRICE_USD_AMT"].dtype == "float64"
    assert frame["ANNUALIZED_NET_PRICE_USD_AMT"].dtype == "float64"
    assert NUMERIC_FEATURES.issubset(frame.columns)
