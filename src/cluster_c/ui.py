"""Small, testable helpers used by the Streamlit decision view."""

from __future__ import annotations

from typing import Any

import pandas as pd


def filter_options(series: pd.Series) -> list[Any]:
    """Return non-null unique filter values in a mixed-type-safe order."""
    values = series.dropna().unique().tolist()
    return sorted(values, key=lambda value: (str(value).casefold(), type(value).__name__))
