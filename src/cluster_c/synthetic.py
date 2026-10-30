"""Deterministic synthetic data matching the supplied Cisco column names."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import INPUT_PATH, RANDOM_SEED, SOURCE_DATASET_VERSION


TERM_LABELS = {1: "1 Year", 3: "3 Year", 5: "5 Year"}
SUITES = {
    "Networking": ("Infrastructure", "Campus Networking"),
    "Security": ("Security", "Zero Trust"),
    "Collaboration": ("Collaboration", "Webex"),
    "Applications": ("Observability", "Application Performance"),
}


def generate_term_deals(seed: int = RANDOM_SEED, row_count: int = 5000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    customer_count = 700
    customer_sizes = rng.lognormal(mean=0.0, sigma=0.55, size=customer_count)
    for index in range(row_count):
        customer_number = int(rng.integers(1, customer_count + 1))
        customer_factor = float(customer_sizes[customer_number - 1])
        start = pd.Timestamp("2020-01-01") + pd.Timedelta(days=int(rng.integers(0, 6 * 365)))
        start = start.normalize()
        suite = str(rng.choice(list(SUITES), p=[0.36, 0.31, 0.19, 0.14]))
        sales_level = str(rng.choice(["Americas", "EMEA", "APJC"], p=[0.44, 0.33, 0.23]))
        ea_category = str(rng.choice(["Full Commit", "Partial Commit"], p=[0.62, 0.38]))
        monetization = str(rng.choice(["Subscription", "Term License"], p=[0.88, 0.12]))
        scms = str(rng.choice(["Enterprise", "Commercial", "Public Sector"], p=[0.42, 0.38, 0.20]))

        # Term selection is intentionally overlapping but observationally confounded.
        size_score = np.log1p(customer_factor)
        probabilities = np.array(
            [0.23 + 0.05 * (scms == "Public Sector"), 0.57, 0.20 + 0.07 * (size_score > 0.8)]
        )
        probabilities = probabilities / probabilities.sum()
        term_years = int(rng.choice([1, 3, 5], p=probabilities))
        end = start + pd.DateOffset(years=term_years) - pd.Timedelta(days=1)

        suite_factor = {"Networking": 1.30, "Security": 1.15, "Collaboration": 0.82, "Applications": 0.95}[suite]
        segment_factor = {"Enterprise": 1.40, "Commercial": 0.78, "Public Sector": 1.08}[scms]
        annual_base = 115_000 * customer_factor * suite_factor * segment_factor
        annual_base *= 1.12 if ea_category == "Full Commit" else 0.86
        annual_base *= 1 + 0.035 * (start.year - 2020)
        term_annual_factor = {1: 0.94, 3: 1.00, 5: 1.075}[term_years]
        annualized = max(10_000, annual_base * term_annual_factor * rng.lognormal(0, 0.10))
        total_commitment = annualized * term_years

        fiscal_year = start.year + 1 if start.month >= 8 else start.year
        fiscal_month = (start.month - 8) % 12 + 1
        fiscal_quarter = (fiscal_month - 1) // 3 + 1
        business_entity, sub_business = SUITES[suite]
        rows.append(
            {
                "MONETIZATION_TYPE_CD": monetization,
                "TERM_BUCKET": TERM_LABELS[term_years],
                "SUBSCRIPTION_REFERENCE_ID": f"SUB-{index + 1:08d}",
                "CAV_END_CUSTOMER_BU_ID": customer_number,
                "CAV_END_CUSTOMER_BU_NAME": f"Synthetic Customer {customer_number:05d}",
                "UCD_C4_ACCOUNT_ID": 100_000 + customer_number,
                "TERM_START_DATE": start,
                "TERM_END_DATE": end,
                "FISCAL_YEAR": fiscal_year,
                "FISCAL_QUARTER_ID": fiscal_quarter,
                "FISCAL_PERIOD_ID": fiscal_month,
                "BOOKINGS_NET_PRICE_USD_AMT": round(total_commitment, 2),
                "ANNUALIZED_NET_PRICE_USD_AMT": round(annualized, 2),
                "SCMS": scms,
                "SALES_LEVEL_2": sales_level,
                "EA_CATEGORY": ea_category,
                "SUITE_NAME": suite,
                "BUSINESS_ENTITY_DESCR": business_entity,
                "SUB_BUSINESS_ENTITY_DESCR": sub_business,
                "DATASET_VERSION": SOURCE_DATASET_VERSION,
            }
        )
    return pd.DataFrame(rows).sort_values("SUBSCRIPTION_REFERENCE_ID").reset_index(drop=True)


def write_term_deals(frame: pd.DataFrame, path=INPUT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, date_format="%Y-%m-%d")
