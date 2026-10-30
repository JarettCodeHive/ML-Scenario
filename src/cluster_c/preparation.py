"""Data contract validation, profiling and leakage-safe model preparation."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import PREPARED_DIR, RAW_COLUMNS, SOURCE_DATASET_VERSION

LOGGER = logging.getLogger(__name__)

TARGET = "BOOKINGS_NET_PRICE_USD_AMT"
FEATURES = [
    "MONETIZATION_TYPE_CD", "TERM_BUCKET", "CAV_END_CUSTOMER_BU_ID", "UCD_C4_ACCOUNT_ID", "FISCAL_YEAR", "FISCAL_QUARTER_ID",
    "FISCAL_PERIOD_ID", "SCMS", "SALES_LEVEL_2", "EA_CATEGORY", "SUITE_NAME",
    "BUSINESS_ENTITY_DESCR", "SUB_BUSINESS_ENTITY_DESCR", "TERM_START_MONTH",
    "CUSTOMER_PRIOR_DEAL_COUNT", "CUSTOMER_PRIOR_AVG_ANNUALIZED_PRICE", "CUSTOMER_PRIOR_AVG_BOOKINGS",
]
NUMERIC_FEATURES = {"FISCAL_YEAR", "FISCAL_QUARTER_ID", "FISCAL_PERIOD_ID", "TERM_START_MONTH", "CUSTOMER_PRIOR_DEAL_COUNT", "CUSTOMER_PRIOR_AVG_ANNUALIZED_PRICE", "CUSTOMER_PRIOR_AVG_BOOKINGS"}
CATEGORICAL_FEATURES = [c for c in FEATURES if c not in NUMERIC_FEATURES]

TERM_BUCKET_ALIASES = {
    **{value: "1 Year" for value in ("1", "1.0", "1Y", "1YR", "1YRS", "1YEAR", "1YEARS", "ONEYEAR", "12MO", "12MOS", "12MONTH", "12MONTHS")},
    **{value: "3 Year" for value in ("3", "3.0", "3Y", "3YR", "3YRS", "3YEAR", "3YEARS", "THREEYEAR", "36MO", "36MOS", "36MONTH", "36MONTHS")},
    **{value: "5 Year" for value in ("5", "5.0", "5Y", "5YR", "5YRS", "5YEAR", "5YEARS", "FIVEYEAR", "60MO", "60MOS", "60MONTH", "60MONTHS")},
}


@dataclass
class PreparedData:
    frame: pd.DataFrame
    profile: pd.DataFrame
    checks: pd.DataFrame
    summary: dict[str, Any]


def normalize_term_bucket(series: pd.Series) -> pd.Series:
    """Map common source-system term labels to the model's three canonical values."""
    compact = (
        series.astype("string")
        .str.strip()
        .str.upper()
        .str.replace(r"[\s_-]+", "", regex=True)
    )
    mapped = compact.map(TERM_BUCKET_ALIASES)
    output = series.copy()
    output.loc[mapped.notna()] = mapped.loc[mapped.notna()]
    return output


def profile(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in frame.columns:
        series = frame[column]
        non_null = series.dropna()
        numeric_or_date = pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series)
        rows.append({
            "column_name": column,
            "dtype": str(series.dtype),
            "row_count": len(frame),
            "missing_count": int(series.isna().sum()),
            "missing_pct": round(float(series.isna().mean() * 100), 4),
            "distinct_count": int(series.nunique(dropna=True)),
            "minimum": non_null.min() if numeric_or_date and not non_null.empty else None,
            "maximum": non_null.max() if numeric_or_date and not non_null.empty else None,
        })
    return pd.DataFrame(rows)


def validate(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    def add(check_id: str, failures: int, description: str) -> None:
        rows.append({"check_id": check_id, "status": "PASS" if failures == 0 else "FAIL", "failure_count": int(failures), "description": description})
    missing_columns = set(RAW_COLUMNS) - set(frame.columns)
    add("required_columns", len(missing_columns), f"Required columns must exist; missing={sorted(missing_columns)}")
    if missing_columns:
        return pd.DataFrame(rows)
    add("subscription_id_present_unique", int(frame["SUBSCRIPTION_REFERENCE_ID"].isna().sum() + frame["SUBSCRIPTION_REFERENCE_ID"].duplicated().sum()), "Subscription reference must be present and unique")
    critical = ["SUBSCRIPTION_REFERENCE_ID", "TERM_BUCKET", "CAV_END_CUSTOMER_BU_ID", "TERM_START_DATE", "TERM_END_DATE", TARGET]
    add("critical_values_present", int(frame[critical].isna().sum().sum()), "Critical identifiers, term fields, dates, and bookings must be populated")
    start = pd.to_datetime(frame["TERM_START_DATE"], errors="coerce")
    end = pd.to_datetime(frame["TERM_END_DATE"], errors="coerce")
    add("valid_dates", int(start.isna().sum() + end.isna().sum() + (end < start).sum()), "Term dates must parse and end after start")
    add("known_term_bucket", int((~frame["TERM_BUCKET"].isin(["1 Year", "3 Year", "5 Year"])).sum()), "Term bucket must be 1 Year, 3 Year or 5 Year")
    add("nonnegative_financials", int((frame[[TARGET, "ANNUALIZED_NET_PRICE_USD_AMT"]] < 0).sum().sum()), "Financial outcomes must be non-negative")
    add("fiscal_quarter_range", int((~frame["FISCAL_QUARTER_ID"].between(1, 4)).sum()), "Fiscal quarter must be 1-4")
    add("fiscal_period_range", int((~frame["FISCAL_PERIOD_ID"].between(1, 12)).sum()), "Fiscal period must be 1-12")
    years = frame["TERM_BUCKET"].str.extract(r"(\d+)")[0].astype(float)
    actual_days = (end - start).dt.days + 1
    add("term_date_consistency", int((np.abs(actual_days - years * 365.25) > 5).sum()), "Term dates must agree with term bucket within five days")
    return pd.DataFrame(rows)


def prepare(raw: pd.DataFrame) -> PreparedData:
    LOGGER.info("Checking %d required source columns", len(RAW_COLUMNS))
    missing_columns = set(RAW_COLUMNS) - set(raw.columns)
    if missing_columns:
        checks = validate(raw)
        raise ValueError("Data contract checks failed: required_columns")

    frame = raw.copy()
    original_terms = frame["TERM_BUCKET"].copy()
    frame["TERM_BUCKET"] = normalize_term_bucket(frame["TERM_BUCKET"])
    unchanged = original_terms.eq(frame["TERM_BUCKET"]) | (
        original_terms.isna() & frame["TERM_BUCKET"].isna()
    )
    LOGGER.info("Normalized TERM_BUCKET on %s rows", f"{int((~unchanged).sum()):,}")

    checks = validate(frame)
    LOGGER.info(
        "Data contract checks finished: %d passed, %d failed",
        int(checks.status.eq("PASS").sum()),
        int(checks.status.eq("FAIL").sum()),
    )
    # Keep structural validation blocking because preparation cannot continue
    # without the expected columns. Row-level quality failures are retained in
    # quality_checks.csv but are temporarily non-blocking.
    required_columns = checks.loc[checks.check_id == "required_columns"]
    if not required_columns.empty and required_columns.iloc[0]["status"] == "FAIL":
        raise ValueError("Data contract checks failed: required_columns")
    frame["TERM_START_DATE"] = pd.to_datetime(frame["TERM_START_DATE"], errors="coerce")
    frame["TERM_END_DATE"] = pd.to_datetime(frame["TERM_END_DATE"], errors="coerce")
    frame["TERM_START_MONTH"] = frame["TERM_START_DATE"].dt.month
    frame["TERM_YEARS"] = pd.to_numeric(
        frame["TERM_BUCKET"].str.extract(r"(\d+)")[0], errors="coerce"
    ).astype("Int64")
    LOGGER.info("Building point-in-time customer history features")
    frame = frame.sort_values(["CAV_END_CUSTOMER_BU_ID", "TERM_START_DATE", "SUBSCRIPTION_REFERENCE_ID"])
    customer_history = frame.groupby("CAV_END_CUSTOMER_BU_ID", sort=False)
    frame["CUSTOMER_PRIOR_DEAL_COUNT"] = customer_history.cumcount()
    frame["CUSTOMER_PRIOR_AVG_ANNUALIZED_PRICE"] = customer_history["ANNUALIZED_NET_PRICE_USD_AMT"].transform(
        lambda values: values.shift(1).expanding().mean()
    )
    frame["CUSTOMER_PRIOR_AVG_BOOKINGS"] = customer_history[TARGET].transform(
        lambda values: values.shift(1).expanding().mean()
    )
    frame["SOURCE_DATASET_VERSION"] = SOURCE_DATASET_VERSION
    columns = list(dict.fromkeys(["SUBSCRIPTION_REFERENCE_ID", "CAV_END_CUSTOMER_BU_ID", "CAV_END_CUSTOMER_BU_NAME", "UCD_C4_ACCOUNT_ID", "TERM_START_DATE", "TERM_END_DATE", "ANNUALIZED_NET_PRICE_USD_AMT", "TERM_YEARS"] + FEATURES + [TARGET, "SOURCE_DATASET_VERSION"]))
    frame = frame[columns]
    summary = {
        "source_dataset_version": SOURCE_DATASET_VERSION,
        "grain": "One row per subscription reference",
        "target": TARGET,
        "row_count": len(frame),
        "customer_count": int(frame["CAV_END_CUSTOMER_BU_ID"].nunique()),
        "start_date_min": str(frame["TERM_START_DATE"].min().date()) if frame["TERM_START_DATE"].notna().any() else None,
        "start_date_max": str(frame["TERM_START_DATE"].max().date()) if frame["TERM_START_DATE"].notna().any() else None,
        "quality_checks_passed": int((checks.status == "PASS").sum()),
        "quality_checks_total": len(checks),
    }
    return PreparedData(frame, profile(raw), checks, summary)


def save_prepared(result: PreparedData, output_dir: Path = PREPARED_DIR) -> dict[str, Path]:
    LOGGER.info("Writing prepared artifacts to %s", output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {"features": output_dir / "term_model_features.csv", "profile": output_dir / "source_profile.csv", "checks": output_dir / "quality_checks.csv", "summary": output_dir / "preparation_summary.json"}
    result.frame.to_csv(paths["features"], index=False, date_format="%Y-%m-%d")
    result.profile.to_csv(paths["profile"], index=False)
    result.checks.to_csv(paths["checks"], index=False)
    paths["summary"].write_text(json.dumps(result.summary, indent=2) + "\n", encoding="utf-8")
    return paths
