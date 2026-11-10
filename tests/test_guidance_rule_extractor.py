import pandas as pd

from cluster_c.unstructured.preparation import prepare_guidance
from cluster_c.unstructured.rule_extractor import extract_candidate_rules


def test_controlled_sample_extracts_six_candidate_rules() -> None:
    prepared = prepare_guidance()
    extracted = extract_candidate_rules(prepared.content)
    rules = {rule.rule_id: rule for rule in extracted.rules}

    assert extracted.is_valid
    assert set(rules) == {
        "GROWTH-G-001",
        "GROWTH-G-002",
        "CUSTOMER-CV-001",
        "CUSTOMER-CV-002",
        "SIMPLICITY-S-001",
        "SIMPLICITY-S-002",
    }
    assert rules["GROWTH-G-001"].operator == "greater_than"
    assert rules["GROWTH-G-001"].threshold == 0
    assert rules["CUSTOMER-CV-001"].operator == "report_only"
    assert rules["SIMPLICITY-S-001"].measure == "scenario_renewal_points"
    assert rules["SIMPLICITY-S-001"].comparison_measure == "baseline_renewal_points"
    assert all(rule.review_status == "not_approved" for rule in rules.values())


def test_three_outcome_source_language_requires_review() -> None:
    prepared = prepare_guidance()
    extracted = extract_candidate_rules(prepared.content)
    warnings = extracted.checks.loc[extracted.checks.status == "WARN"]

    assert warnings["check_id"].tolist() == ["multi_outcome_rule_requires_review"]
    assert "GROWTH-G-001" in warnings.iloc[0].detail


def scoped_page(scope: str) -> pd.DataFrame:
    prepared = prepare_guidance()
    source = prepared.content.iloc[0]
    return pd.DataFrame([{
        **source.to_dict(),
        "page_number": 10,
        "page_text": "\n".join([
            "SAMPLE RULE DEFINITIONS",
            "GROWTH-G-SCOPED - Scoped commitment direction",
            f"Scope: {scope}",
            "Measure: commitment_change_pct",
            "Favorable: value greater than 0 percent.",
            "Attention: value less than or equal to 0 percent.",
            "Explanation: scoped sample guidance.",
        ]),
    }])


def test_candidate_extractor_reads_optional_segment_scope() -> None:
    result = extract_candidate_rules(scoped_page("SCMS=ENTERPRISE"))

    assert result.is_valid
    assert len(result.rules) == 1
    rule = result.rules[0]
    assert rule.scope_type == "portfolio_filter"
    assert rule.scope_column == "SCMS"
    assert rule.scope_value == "ENTERPRISE"
    assert result.checks.loc[result.checks.check_id == "scope_valid", "status"].eq("PASS").all()


def test_candidate_extractor_rejects_unapproved_scope_column() -> None:
    result = extract_candidate_rules(scoped_page("SUBSCRIPTION_REFERENCE_ID=SR1"))

    assert not result.is_valid
    assert result.rules == []
    failure = result.checks.loc[result.checks.check_id == "candidate_rule_schema_valid"].iloc[0]
    assert failure.status == "FAIL"
    assert "scope_column" in failure.detail
