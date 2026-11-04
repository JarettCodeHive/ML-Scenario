"""Reusable Streamlit rendering for evaluated EA guidance."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from .ui import guidance_panel_state, guidance_status_label
from .unstructured.decision_rules import GuidanceRegistry


def render_lens_guidance(evaluation: dict, lens: str) -> None:
    """Render the status and reviewed explanations for one decision lens."""
    result = evaluation["lenses"][lens]
    st.markdown(f"**EA guidance: {guidance_status_label(result['status'])}**")
    for reason in result["reasons"]:
        st.caption(reason)


def render_guidance_traceability(
    guidance_error: str | None,
    guidance_registry: GuidanceRegistry | None,
    guidance_evaluation: dict,
) -> None:
    """Render guidance availability or full rule-level provenance."""
    with st.expander("EA guidance applied"):
        guidance_panel, guidance_panel_message = guidance_panel_state(
            guidance_error,
            len(guidance_registry.candidate_rules) if guidance_registry else 0,
            len(guidance_registry.active_rules) if guidance_registry else 0,
            guidance_evaluation["applied_rule_count"],
        )
        if guidance_panel == "error":
            st.warning(guidance_panel_message)
            st.code("PYTHONPATH=src python scripts/prepare_ea_guidance.py", language="bash")
        elif guidance_panel == "unapproved":
            st.info(guidance_panel_message)
            if guidance_registry:
                st.caption(
                    f"Registry generated {guidance_registry.generated_at_utc}; "
                    f"{len(guidance_registry.candidate_rules)} candidate rules and 0 active rules."
                )
        elif guidance_panel == "out_of_scope":
            st.info(guidance_panel_message)
            st.caption(
                f"Registry generated {guidance_registry.generated_at_utc}; "
                f"{len(guidance_registry.active_rules)} active rules."
            )
        else:
            guidance_rows = pd.DataFrame(guidance_evaluation["rule_results"])
            guidance_rows["source_fingerprint"] = guidance_rows["source_sha256"].str.slice(0, 12)
            st.dataframe(
                guidance_rows[[
                    "lens", "status", "rule_id", "reason", "measure", "measure_value",
                    "comparison_measure", "comparison_value", "operator", "threshold",
                    "source_id", "source_version", "source_page", "source_fingerprint",
                ]],
                width="stretch",
                hide_index=True,
            )
            st.caption(
                f"Applied {guidance_evaluation['applied_rule_count']} approved rules from registry "
                f"generated {guidance_registry.generated_at_utc}."
            )
