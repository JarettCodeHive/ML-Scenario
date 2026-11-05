import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from cluster_c.config import GUIDANCE_MANIFEST_PATH
from cluster_c.unstructured.approval import APPROVAL_COLUMNS, apply_human_approvals
from cluster_c.unstructured.decision_rules import (
    GuidanceRegistryError,
    evaluate_guidance,
    load_guidance_registry,
)
from cluster_c.unstructured.schemas import ALLOWED_OPERATORS, GuidanceRule


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "guidance"
APPROVED_REGISTRY = FIXTURE_DIR / "approved_registry.json"
LIFECYCLE_CASES = FIXTURE_DIR / "lifecycle_cases.json"
MALFORMED_REGISTRY = FIXTURE_DIR / "malformed_registry.json"


def _approved_rules():
    return load_guidance_registry(APPROVED_REGISTRY).active_rules


def _by_rule(result):
    return {row["rule_id"]: row for row in result["rule_results"]}


def test_operational_poc_source_is_registered_for_direct_activation() -> None:
    manifest = pd.read_csv(GUIDANCE_MANIFEST_PATH, dtype="string", keep_default_na=False)
    source = manifest.loc[manifest.source_id.eq("EA-GUIDE-SAMPLE-001")]

    assert len(source) == 1
    assert source.iloc[0].is_synthetic.casefold() == "false"
    assert all(rule.source_id.startswith("EA-TEST-") for rule in _approved_rules())


def test_approved_fixture_covers_every_operator_and_scope_type() -> None:
    rules = _approved_rules()

    assert {rule.operator for rule in rules} == ALLOWED_OPERATORS
    assert {rule.scope_type for rule in rules} == {"global", "portfolio_filter"}
    assert {rule.scope_column for rule in rules if rule.scope_type == "portfolio_filter"} == {
        "SCMS",
        "SALES_LEVEL_2",
        "SUITE_NAME",
    }


def test_fixture_produces_favorable_attention_neutral_and_report_only_outcomes() -> None:
    result = evaluate_guidance(
        {
            "commitment_change_pct": 5.0,
            "npv_change": -1.0,
            "flexible_adoption_pct": 90.0,
            "scenario_renewal_points": 10.0,
            "baseline_renewal_points": 10.0,
            "average_term_years": 3.0,
            "discount_rate_pct": 8.0,
        },
        {},
        _approved_rules(),
        {
            "SCMS": ["ENTERPRISE"],
            "SALES_LEVEL_2": ["EMEA_NORTH"],
            "SUITE_NAME": ["Cisco DNA Wireless"],
        },
    )
    rules = _by_rule(result)

    assert rules["ACCEPT-GLOBAL-GT"]["status"] == "favorable"
    assert rules["ACCEPT-GLOBAL-GTE"]["status"] == "neutral"
    assert rules["ACCEPT-SCMS-LT"]["status"] == "attention"
    assert rules["ACCEPT-SALES-LTE"]["status"] == "favorable"
    assert rules["ACCEPT-SUITE-EQ"]["status"] == "favorable"
    assert rules["ACCEPT-GLOBAL-REPORT"]["status"] == "report_only"


@pytest.mark.parametrize(
    ("filter_context", "expected_rule_id"),
    [
        ({"SCMS": ["ENTERPRISE"]}, "ACCEPT-SCMS-LT"),
        ({"SALES_LEVEL_2": ["EMEA_NORTH"]}, "ACCEPT-SALES-LTE"),
        ({"SUITE_NAME": ["Cisco DNA Wireless"]}, "ACCEPT-SUITE-EQ"),
    ],
)
def test_each_segment_scope_applies_only_to_an_exact_filter(filter_context, expected_rule_id) -> None:
    values = {
        "commitment_change_pct": 1.0,
        "npv_change": 1.0,
        "flexible_adoption_pct": 70.0,
        "scenario_renewal_points": 9.0,
        "average_term_years": 3.0,
        "discount_rate_pct": 8.0,
    }

    result = evaluate_guidance(values, {}, _approved_rules(), filter_context)
    applied_ids = {row["rule_id"] for row in result["rule_results"]}

    assert expected_rule_id in applied_ids
    assert {"ACCEPT-GLOBAL-GT", "ACCEPT-GLOBAL-GTE", "ACCEPT-GLOBAL-REPORT"}.issubset(applied_ids)


def test_global_rules_apply_without_filters_and_scoped_rules_do_not() -> None:
    result = evaluate_guidance(
        {"commitment_change_pct": 1, "npv_change": 1, "discount_rate_pct": 8},
        {},
        _approved_rules(),
    )

    assert {row["rule_id"] for row in result["rule_results"]} == {
        "ACCEPT-GLOBAL-GT",
        "ACCEPT-GLOBAL-GTE",
        "ACCEPT-GLOBAL-REPORT",
    }


def test_mixed_segment_filter_does_not_apply_scoped_rule() -> None:
    result = evaluate_guidance(
        {"commitment_change_pct": 1, "npv_change": 1, "discount_rate_pct": 8},
        {},
        _approved_rules(),
        {"SCMS": ["ENTERPRISE", "COMMERCIAL"]},
    )

    assert "ACCEPT-SCMS-LT" not in {row["rule_id"] for row in result["rule_results"]}


def test_missing_measure_returns_unknown_without_hiding_other_results() -> None:
    result = evaluate_guidance(
        {"npv_change": 1.0, "discount_rate_pct": 8.0},
        {},
        _approved_rules(),
    )
    rules = _by_rule(result)

    assert rules["ACCEPT-GLOBAL-GT"]["status"] == "unknown"
    assert rules["ACCEPT-GLOBAL-GTE"]["status"] == "favorable"
    assert rules["ACCEPT-GLOBAL-REPORT"]["status"] == "report_only"


def test_lifecycle_fixture_activates_only_current_hash_matched_approval(tmp_path) -> None:
    payload = json.loads(LIFECYCLE_CASES.read_text(encoding="utf-8"))
    content = pd.DataFrame(payload["content"])
    rules = [GuidanceRule.from_mapping(values) for values in payload["rules"]]
    approvals_path = tmp_path / "approvals.csv"
    pd.DataFrame(payload["approvals"], columns=APPROVAL_COLUMNS).to_csv(approvals_path, index=False)

    result = apply_human_approvals(
        rules,
        content,
        approvals_path=approvals_path,
        as_of=date.fromisoformat(payload["as_of"]),
    )

    assert result.is_valid
    assert [rule.rule_id for rule in result.active_rules] == payload["expected_active_rule_ids"]
    statuses = {rule.rule_id: rule.review_status for rule in result.candidate_rules}
    assert statuses["ACCEPT-LIFECYCLE-REJECTED"] == "rejected"
    assert statuses["ACCEPT-LIFECYCLE-FUTURE"] == "approved"
    assert statuses["ACCEPT-LIFECYCLE-RETIRED"] == "approved"
    assert statuses["ACCEPT-LIFECYCLE-HASH"] == "not_approved"


def test_no_approved_guidance_is_explicitly_unavailable() -> None:
    result = evaluate_guidance({}, {}, [])

    assert not result["applied"]
    assert result["active_rule_count"] == 0
    assert all(lens["status"] == "not_available" for lens in result["lenses"].values())


def test_missing_and_malformed_registries_fail_safely(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_guidance_registry(tmp_path / "missing.json")
    with pytest.raises(GuidanceRegistryError, match="Unable to read"):
        load_guidance_registry(MALFORMED_REGISTRY)
