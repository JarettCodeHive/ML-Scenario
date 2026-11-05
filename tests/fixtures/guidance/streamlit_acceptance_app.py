"""Small Streamlit harness exercising the production EA-guidance components."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from cluster_c.guidance_streamlit import render_guidance_traceability, render_lens_guidance  # noqa: E402
from cluster_c.ui import apply_portfolio_filters  # noqa: E402
from cluster_c.unstructured.decision_rules import (  # noqa: E402
    GuidanceRegistry,
    evaluate_guidance,
    load_guidance_registry,
)


FIXTURE_REGISTRY = Path(__file__).with_name("approved_registry.json")
approved_registry = load_guidance_registry(FIXTURE_REGISTRY)

st.set_page_config(page_title="EA Guidance Acceptance Harness", layout="wide")
st.title("EA Guidance Streamlit Acceptance")

mode = st.selectbox(
    "Guidance state",
    ["Approved fixture", "No approved guidance", "Missing registry"],
)
commitment_change_pct = st.slider("Commitment change percent", -10, 10, 5)
npv_change = st.slider("NPV change", -10, 10, 1)
flexible_adoption_pct = st.slider("Flexible adoption percent", 0, 100, 70)
scenario_renewal_points = st.slider("Scenario renewal points", 0, 20, 9)
average_term_years = st.slider("Average term years", 1, 5, 3)
sales_levels = st.multiselect("Sales level 2", ["EMEA_NORTH", "US COMMERCIAL", "NO_MATCH"])
segments = st.multiselect("SCMS", ["ENTERPRISE", "COMMERCIAL", "NO_MATCH"])
suites = st.multiselect("Suite", ["Cisco DNA Wireless", "Duo", "NO_MATCH"])

st.metric("Structured scenario commitment", "$1,000,000")
portfolio = pd.DataFrame({
    "SALES_LEVEL_2": ["EMEA_NORTH", "US COMMERCIAL"],
    "SCMS": ["ENTERPRISE", "COMMERCIAL"],
    "SUITE_NAME": ["Cisco DNA Wireless", "Duo"],
})
scope = apply_portfolio_filters(portfolio, sales_levels, segments, suites)
if scope.empty:
    st.info("No rows match the selected scope.")

if mode == "Approved fixture":
    registry = approved_registry
    guidance_error = None
elif mode == "No approved guidance":
    registry = GuidanceRegistry(
        schema_version=approved_registry.schema_version,
        generated_at_utc=approved_registry.generated_at_utc,
        candidate_rules=approved_registry.candidate_rules,
        active_rules=[],
    )
    guidance_error = None
else:
    registry = None
    guidance_error = "Acceptance fixture registry is missing"

evaluation = evaluate_guidance(
    {
        "commitment_change_pct": commitment_change_pct,
        "npv_change": npv_change,
        "flexible_adoption_pct": flexible_adoption_pct,
        "scenario_renewal_points": scenario_renewal_points,
        "baseline_renewal_points": 10,
        "average_term_years": average_term_years,
        "discount_rate_pct": 8,
    },
    {},
    registry.active_rules if registry else [],
    {
        "SALES_LEVEL_2": sales_levels,
        "SCMS": segments,
        "SUITE_NAME": suites,
    },
)

st.subheader("Outcome on the three decision lenses")
growth, customer, simplicity = st.columns(3)
with growth.container(border=True):
    st.markdown("#### 1. Growth")
    render_lens_guidance(evaluation, "growth")
with customer.container(border=True):
    st.markdown("#### 2. Customer value")
    render_lens_guidance(evaluation, "customer_value")
with simplicity.container(border=True):
    st.markdown("#### 3. Simplicity with scale")
    render_lens_guidance(evaluation, "simplicity_with_scale")

render_guidance_traceability(guidance_error, registry, evaluation)
