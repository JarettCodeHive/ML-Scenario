"""Transparent Cluster C portfolio scenario calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _npv_stream(total: pd.Series, years: int, discount_rate: float, horizon: int = 5) -> pd.Series:
    annual = total / years
    factor = sum(1 / ((1 + discount_rate) ** year) for year in range(1, min(years, horizon) + 1))
    return annual * factor


def calculate_scenario(
    scored: pd.DataFrame,
    flexible_adoption_pct: float = 100,
    five_year_share_pct: float = 50,
    discount_rate_pct: float = 8,
) -> pd.DataFrame:
    if not 0 <= flexible_adoption_pct <= 100 or not 0 <= five_year_share_pct <= 100:
        raise ValueError("Scenario shares must be between 0 and 100")
    if discount_rate_pct < 0:
        raise ValueError("Discount rate must be non-negative")
    result = scored.copy()
    adoption = flexible_adoption_pct / 100
    five_share = five_year_share_pct / 100
    one_share = 1 - five_share
    rate = discount_rate_pct / 100
    result["baseline_3y_commitment"] = result["bookings_3y"]
    result["scenario_expected_commitment"] = (1 - adoption) * result["bookings_3y"] + adoption * (one_share * result["bookings_1y"] + five_share * result["bookings_5y"])
    result["scenario_commitment_p10"] = (1 - adoption) * result["bookings_3y_p10"] + adoption * (one_share * result["bookings_1y_p10"] + five_share * result["bookings_5y_p10"])
    result["scenario_commitment_p90"] = (1 - adoption) * result["bookings_3y_p90"] + adoption * (one_share * result["bookings_1y_p90"] + five_share * result["bookings_5y_p90"])
    result["commitment_change"] = result["scenario_expected_commitment"] - result["baseline_3y_commitment"]
    result["expected_term_years"] = (1 - adoption) * 3 + adoption * (one_share * 1 + five_share * 5)
    result["baseline_npv"] = _npv_stream(result["bookings_3y"], 3, rate)
    result["scenario_npv"] = (1 - adoption) * _npv_stream(result["bookings_3y"], 3, rate) + adoption * (one_share * _npv_stream(result["bookings_1y"], 1, rate) + five_share * _npv_stream(result["bookings_5y"], 5, rate))
    result["npv_change"] = result["scenario_npv"] - result["baseline_npv"]
    result["baseline_renewal_decision_points_5y"] = 1.0
    result["scenario_renewal_decision_points_5y"] = (1 - adoption) + adoption * (one_share * 5 + five_share)
    return result


def annual_coverage(details: pd.DataFrame, adoption_pct: float, five_year_share_pct: float) -> pd.DataFrame:
    adoption = adoption_pct / 100; five_share = five_year_share_pct / 100; one_share = 1 - five_share
    baseline_annual = float((details["bookings_3y"] / 3).sum())
    scenario_3 = float(((1 - adoption) * details["bookings_3y"] / 3).sum())
    scenario_1 = float((adoption * one_share * details["bookings_1y"]).sum())
    scenario_5 = float((adoption * five_share * details["bookings_5y"] / 5).sum())
    return pd.DataFrame({
        "portfolio_year": range(1, 6),
        "baseline_3y_coverage": [baseline_annual, baseline_annual, baseline_annual, 0, 0],
        "flexible_scenario_coverage": [scenario_3 + scenario_1 + scenario_5, scenario_3 + scenario_5, scenario_3 + scenario_5, scenario_5, scenario_5],
    })


def summarise(details: pd.DataFrame, coverage: pd.DataFrame) -> dict[str, float | int]:
    baseline = float(details["baseline_3y_commitment"].sum()); scenario = float(details["scenario_expected_commitment"].sum())
    baseline_cov = coverage["baseline_3y_coverage"].to_numpy(float); scenario_cov = coverage["flexible_scenario_coverage"].to_numpy(float)
    cv = lambda values: float(np.std(values) / np.mean(values) * 100) if np.mean(values) else float("nan")
    return {
        "deals": len(details),
        "baseline_commitment": baseline,
        "scenario_commitment": scenario,
        "commitment_change": scenario - baseline,
        "commitment_change_pct": (scenario - baseline) / baseline * 100 if baseline else 0,
        "scenario_p10": float(details["scenario_commitment_p10"].sum()),
        "scenario_p90": float(details["scenario_commitment_p90"].sum()),
        "baseline_npv": float(details["baseline_npv"].sum()),
        "scenario_npv": float(details["scenario_npv"].sum()),
        "npv_change": float(details["npv_change"].sum()),
        "average_term_years": float(details["expected_term_years"].mean()),
        "baseline_coverage_cv_pct": cv(baseline_cov),
        "scenario_coverage_cv_pct": cv(scenario_cov),
        "baseline_renewal_points": float(details["baseline_renewal_decision_points_5y"].sum()),
        "scenario_renewal_points": float(details["scenario_renewal_decision_points_5y"].sum()),
    }


def sensitivity(scored: pd.DataFrame, adoption_pct: float, discount_rate_pct: float) -> pd.DataFrame:
    rows = []
    for share in range(0, 101, 5):
        details = calculate_scenario(scored, adoption_pct, share, discount_rate_pct)
        coverage = annual_coverage(details, adoption_pct, share)
        summary = summarise(details, coverage)
        rows.append({"five_year_share_pct": share, "commitment_change": summary["commitment_change"], "npv_change": summary["npv_change"], "coverage_cv_pct": summary["scenario_coverage_cv_pct"]})
    return pd.DataFrame(rows)


def breakeven_share(sensitivity_frame: pd.DataFrame) -> float | None:
    eligible = sensitivity_frame[sensitivity_frame["npv_change"] >= 0]
    return None if eligible.empty else float(eligible["five_year_share_pct"].min())
