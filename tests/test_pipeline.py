import numpy as np
import pandas as pd

from cluster_c.config import RAW_COLUMNS
from cluster_c.preparation import prepare
from cluster_c.synthetic import generate_term_deals


def test_synthetic_data_is_deterministic_and_matches_contract() -> None:
    first = generate_term_deals(seed=42, row_count=200)
    second = generate_term_deals(seed=42, row_count=200)
    pd.testing.assert_frame_equal(first, second)
    assert set(RAW_COLUMNS).issubset(first.columns)
    assert set(first.TERM_BUCKET) == {"1 Year", "3 Year", "5 Year"}


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
