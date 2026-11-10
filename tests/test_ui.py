import numpy as np
import pandas as pd

from cluster_c.ui import apply_portfolio_filters, filter_options


def test_filter_options_exclude_nulls_and_sort_strings() -> None:
    values = pd.Series(["US COMMERCIAL", np.nan, "EMEA", pd.NA, "AMERICAS_SP"])

    assert filter_options(values) == ["AMERICAS_SP", "EMEA", "US COMMERCIAL"]


def test_filter_options_sort_mixed_types_without_comparing_them_directly() -> None:
    values = pd.Series(["Enterprise", 2.0, None, "Commercial"], dtype="object")

    assert filter_options(values) == [2.0, "Commercial", "Enterprise"]


def test_apply_portfolio_filters_supports_exact_mixed_and_empty_scopes() -> None:
    frame = pd.DataFrame({
        "SALES_LEVEL_2": ["EMEA_NORTH", "US COMMERCIAL"],
        "SCMS": ["ENTERPRISE", "COMMERCIAL"],
        "SUITE_NAME": ["Cisco DNA Wireless", "Duo"],
    })

    exact = apply_portfolio_filters(frame, ["EMEA_NORTH"], ["ENTERPRISE"], ["Cisco DNA Wireless"])
    mixed = apply_portfolio_filters(frame, segments=["ENTERPRISE", "COMMERCIAL"])
    empty = apply_portfolio_filters(frame, suites=["NO_MATCH"])

    assert exact.index.tolist() == [0]
    assert len(mixed) == 2
    assert empty.empty
