"""Standalone Streamlit UI for Cisco EA Cluster C term flexibility."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from cluster_c.model import predict_term_options  # noqa: E402
from cluster_c.scenario import (  # noqa: E402
    annual_coverage,
    breakeven_share,
    calculate_scenario,
    sensitivity,
    summarise,
)

st.set_page_config(page_title="Cisco EA Cluster C ML Scenario", page_icon="📅", layout="wide")


def money(value: float) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.0f}"


@st.cache_data
def load_scored() -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    feature_path = ROOT / "data/prepared/term_model_features.csv"
    model_dir = ROOT / "data/models/cluster_c_term_v1"
    required = [feature_path, model_dir / "metadata.json", model_dir / "metrics.csv"]
    required += [model_dir / name for name in ("bookings_point.cbm", "bookings_lower_10.cbm", "bookings_upper_90.cbm")]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing artifacts: " + ", ".join(missing))
    frame = pd.read_csv(feature_path, parse_dates=["TERM_START_DATE", "TERM_END_DATE"])
    predictions = predict_term_options(frame, model_dir)
    frame = frame.merge(predictions, on="SUBSCRIPTION_REFERENCE_ID", validate="one_to_one")
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(model_dir / "metrics.csv")
    return frame, metadata, metrics


st.title("Cisco EA — Cluster C ML Scenario")
st.markdown(
    """
**What-if scenario:** Flexible 1- or 5-year terms against 3-year  
**Core question:** Does term flexibility grow commitments or destabilise revenue?  
**PPT classification:** Cluster C · Term and Renewal · Wave 1 · Scenario 8 · Type 1
"""
)
st.warning(
    "Synthetic POC. CatBoost estimates bookings associations under each term option; it does not prove "
    "that changing term causes the outcome. Revenue coverage is a straight-line contracted-value proxy, "
    "not GAAP revenue recognition, cash flow, or Cisco-approved accounting logic."
)

try:
    scored, metadata, model_metrics = load_scored()
except (FileNotFoundError, ValueError) as exc:
    st.error(str(exc))
    st.code("PYTHONPATH=src python scripts/prepare_data.py\nPYTHONPATH=src python scripts/train_model.py", language="bash")
    st.stop()

f1, f2, f3 = st.columns(3)
sales_levels = f1.multiselect("Sales level 2", sorted(scored["SALES_LEVEL_2"].unique()), placeholder="All")
segments = f2.multiselect("SCMS", sorted(scored["SCMS"].unique()), placeholder="All")
suites = f3.multiselect("Suite", sorted(scored["SUITE_NAME"].unique()), placeholder="All")

l1, l2, l3 = st.columns(3)
adoption = l1.slider("Deals using flexible terms", 0, 100, 100, 5, format="%d%%")
five_share = l2.slider("Five-year share of flexible deals", 0, 100, 50, 5, format="%d%%")
discount_rate = l3.slider("Annual discount rate", 0, 20, 8, 1, format="%d%%")

scope = scored.copy()
if sales_levels: scope = scope[scope.SALES_LEVEL_2.isin(sales_levels)]
if segments: scope = scope[scope.SCMS.isin(segments)]
if suites: scope = scope[scope.SUITE_NAME.isin(suites)]
if scope.empty:
    st.info("No rows match the selected scope.")
    st.stop()

details = calculate_scenario(scope, adoption, five_share, discount_rate)
coverage = annual_coverage(details, adoption, five_share)
summary = summarise(details, coverage)
sensitivity_frame = sensitivity(scope, adoption, discount_rate)
breakeven = breakeven_share(sensitivity_frame)

st.caption(
    f"Model {metadata['model_version']} · {summary['deals']:,} eligible deals · "
    "3-year reference term · local CatBoost"
)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Scenario expected commitment", money(summary["scenario_commitment"]), delta=f"{summary['commitment_change_pct']:+.1f}% vs 3-year")
m2.metric("Commitment change", money(summary["commitment_change"]))
m3.metric("NPV change", money(summary["npv_change"]), help="Straight-line contracted-value proxy discounted at the selected rate")
m4.metric("Average expected term", f"{summary['average_term_years']:.2f} years", delta=f"{summary['average_term_years'] - 3:+.2f} vs baseline", delta_color="off")

st.subheader("Outcome on the three PPT decision lenses")
g, v, s = st.columns(3)
with g.container(border=True):
    st.markdown("#### 1. Growth")
    st.metric("Expected commitment change", money(summary["commitment_change"]), delta=f"{summary['commitment_change_pct']:+.1f}%")
    st.metric("Experimental P10–P90 scenario range", f"{money(summary['scenario_p10'])} – {money(summary['scenario_p90'])}")
    st.caption("Modelled total bookings commitment; range is not yet calibrated on Cisco data.")
with v.container(border=True):
    st.markdown("#### 2. Customer value")
    st.metric("Flexible deals", f"{adoption}%", delta=f"{100 - five_share}% one-year · {five_share}% five-year", delta_color="off")
    st.metric("Average expected term", f"{summary['average_term_years']:.2f} years")
    st.caption("Choice and term length proxy only. Renewal, churn, satisfaction, and acceptance are not predicted.")
with s.container(border=True):
    st.markdown("#### 3. Simplicity with scale")
    renewal_change = summary["scenario_renewal_points"] - summary["baseline_renewal_points"]
    st.metric("Five-year renewal decision points", f"{summary['scenario_renewal_points']:,.0f}", delta=f"{renewal_change:+,.0f} vs baseline", delta_color="inverse")
    st.metric("Five-year coverage volatility", f"{summary['scenario_coverage_cv_pct']:.1f}%", delta=f"{summary['scenario_coverage_cv_pct'] - summary['baseline_coverage_cv_pct']:+.1f} pts", delta_color="inverse")
    st.caption("Operational and stability proxies; actual approval effort and operating cost are unavailable.")

chart1, chart2 = st.columns(2)
coverage_long = coverage.melt("portfolio_year", var_name="case", value_name="contracted_value_coverage")
chart1.plotly_chart(px.line(coverage_long, x="portfolio_year", y="contracted_value_coverage", color="case", markers=True, title="Five-year contracted-value coverage proxy", labels={"portfolio_year": "Portfolio year", "contracted_value_coverage": "Illustrative USD", "case": ""}), width="stretch")
chart2.plotly_chart(px.line(sensitivity_frame, x="five_year_share_pct", y=["commitment_change", "npv_change"], markers=True, title="Sensitivity to five-year share", labels={"five_year_share_pct": "Five-year share (%)", "value": "Illustrative USD", "variable": "Measure"}), width="stretch")

if breakeven is None:
    st.info("No NPV breakeven is reached across a 0%–100% five-year share under the selected assumptions.")
else:
    st.success(f"NPV breakeven occurs at approximately {breakeven:.0f}% five-year share among flexible deals under the selected assumptions.")

st.subheader("Baseline versus flexible scenario")
comparison = pd.DataFrame({
    "measure": ["Total commitment", "NPV", "Renewal decision points", "Coverage volatility (%)"],
    "3-year baseline": [summary["baseline_commitment"], summary["baseline_npv"], summary["baseline_renewal_points"], summary["baseline_coverage_cv_pct"]],
    "Flexible scenario": [summary["scenario_commitment"], summary["scenario_npv"], summary["scenario_renewal_points"], summary["scenario_coverage_cv_pct"]],
})
st.dataframe(comparison, width="stretch", hide_index=True)

st.subheader("Deal-level counterfactual estimates")
display = details[[
    "SUBSCRIPTION_REFERENCE_ID", "CAV_END_CUSTOMER_BU_NAME", "SCMS", "SALES_LEVEL_2", "SUITE_NAME",
    "bookings_1y", "bookings_3y", "bookings_5y", "scenario_expected_commitment", "commitment_change", "npv_change",
]].copy()
st.dataframe(display, width="stretch", hide_index=True)
st.download_button("Download scenario CSV", display.to_csv(index=False).encode("utf-8"), "cluster_c_term_scenario.csv", "text/csv")

with st.expander("Model performance, overlap, and limitations"):
    st.dataframe(model_metrics[model_metrics.split == "test"], width="stretch", hide_index=True)
    st.markdown(
        f"""
- Test WAPE: **{metadata['test_wape_pct']:.2f}%**.
- Improvement over the term-and-suite median baseline: **{metadata['relative_wape_improvement_pct']:.2f}%**.
- Training requires historical examples of 1-, 3-, and 5-year terms in every chronological split.
- The model is an observational S-learner. Counterfactual estimates require overlap and do not establish causality.
- A production decision needs renewal, churn, acceptance, discount, invoice, cancellation, and operational-cost data listed in the PPT plan.
"""
    )

with st.expander("Traceable calculation flow", expanded=True):
    st.markdown(
        """
1. Validate the supplied Cisco-shaped CSV and preserve one row per subscription.
2. Train CatBoost on historical total bookings with `TERM_BUCKET` as the term-treatment feature.
3. Score every eligible row three times: 1 year, 3 years, and 5 years.
4. Keep all non-term attributes fixed during the three predictions.
5. Build the flexible scenario as a weighted mix of 1-year, 3-year, and 5-year estimates.
6. Spread commitment evenly across the selected term solely for the revenue-stability and NPV proxies.
7. Compare the result with a uniform 3-year reference portfolio.

No SQL is used in this CSV handoff. When Cisco replaces the CSV input with read-only Snowflake, the source query should be displayed here in line with the PPT traceability requirement.
"""
    )
