import numpy as np
import pandas as pd

from cluster_c.ui import filter_options


def test_filter_options_exclude_nulls_and_sort_strings() -> None:
    values = pd.Series(["US COMMERCIAL", np.nan, "EMEA", pd.NA, "AMERICAS_SP"])

    assert filter_options(values) == ["AMERICAS_SP", "EMEA", "US COMMERCIAL"]


def test_filter_options_sort_mixed_types_without_comparing_them_directly() -> None:
    values = pd.Series(["Enterprise", 2.0, None, "Commercial"], dtype="object")

    assert filter_options(values) == [2.0, "Commercial", "Enterprise"]
