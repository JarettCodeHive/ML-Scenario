"""Small, testable helpers used by the Streamlit decision view."""

from __future__ import annotations

from typing import Any

import pandas as pd


GUIDANCE_STATUS_LABELS = {
    "favorable": "Favorable",
    "neutral": "Neutral",
    "attention": "Attention required",
    "report_only": "Informational",
    "unknown": "Unable to evaluate",
    "not_available": "No approved rule applied",
}


def filter_options(series: pd.Series) -> list[Any]:
    """Return non-null unique filter values in a mixed-type-safe order."""
    values = series.dropna().unique().tolist()
    return sorted(values, key=lambda value: (str(value).casefold(), type(value).__name__))


def apply_portfolio_filters(
    frame: pd.DataFrame,
    sales_levels: list[Any] | None = None,
    segments: list[Any] | None = None,
    suites: list[Any] | None = None,
) -> pd.DataFrame:
    """Apply the three supported UI filters without mutating the input frame."""
    scope = frame
    for column, selected in (
        ("SALES_LEVEL_2", sales_levels),
        ("SCMS", segments),
        ("SUITE_NAME", suites),
    ):
        if selected:
            scope = scope.loc[scope[column].isin(selected)]
    return scope


def guidance_status_label(status: str) -> str:
    """Return a stable user-facing label for an evaluated guidance status."""
    return GUIDANCE_STATUS_LABELS.get(status, status.replace("_", " ").title())


def guidance_panel_state(
    error: str | None,
    candidate_count: int,
    active_count: int,
    applied_count: int,
) -> tuple[str, str]:
    """Return the traceability-panel state independently of Streamlit rendering."""
    if error:
        return "error", f"EA guidance is unavailable: {error}"
    if active_count == 0:
        return "unapproved", "No approved EA guidance has been applied. Structured scenario results remain available."
    if applied_count == 0:
        return "out_of_scope", "Approved EA guidance exists, but no rule applies to the currently selected portfolio scope."
    return "applied", f"Applied {applied_count} approved EA guidance rules."
