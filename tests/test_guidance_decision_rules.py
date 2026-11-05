import json
from pathlib import Path

import pytest

from cluster_c.ui import guidance_panel_state, guidance_status_label
from cluster_c.unstructured.decision_rules import (
    GuidanceRegistryError,
    evaluate_guidance,
    load_guidance_registry,
)
from cluster_c.unstructured.schemas import GuidanceRule


SOURCE_HASH = "c1c07722a93f19d9d0f5f886878a01c40f20a7d6dc00c86e1d9d69e33a0ea3da"


def approved_rule(**overrides) -> GuidanceRule:
    values = {
        "rule_id": "GROWTH-G-001",
        "lens": "growth",
        "measure": "commitment_change_pct",
        "operator": "greater_than",
        "threshold": 0.0,
        "comparison_measure": None,
        "status_if_matched": "favorable",
        "status_if_not_matched": "attention",
        "explanation": "Commitment direction follows approved guidance.",
        "scope_type": "global",
        "scope_column": None,
        "scope_value": None,
        "source_id": "EA-GUIDE-APPROVED-001",
        "source_version": "v1",
        "source_page": 2,
        "source_sha256": SOURCE_HASH,
        "review_status": "approved",
    }
    values.update(overrides)
    return GuidanceRule.from_mapping(values)


def test_evaluator_applies_predefined_comparisons_and_aggregates_conservatively() -> None:
    rules = [
        approved_rule(),
        approved_rule(
            rule_id="GROWTH-G-002",
            measure="npv_change",
            operator="greater_than_or_equal",
            explanation="NPV must not decrease.",
        ),
    ]

    result = evaluate_guidance(
        {"commitment_change_pct": 5.0, "npv_change": -1.0},
        {},
        rules,
    )

    assert result["applied"] is True
    assert result["applied_rule_count"] == 2
    assert result["lenses"]["growth"]["status"] == "attention"
    assert [row["status"] for row in result["rule_results"]] == ["favorable", "attention"]


@pytest.mark.parametrize(
    ("operator", "value", "threshold", "expected_status"),
    [
        ("greater_than", 2.0, 1.0, "favorable"),
        ("greater_than_or_equal", 1.0, 1.0, "favorable"),
        ("less_than", 0.0, 1.0, "favorable"),
        ("less_than_or_equal", 1.0, 1.0, "favorable"),
        ("equal", 1.0, 1.0, "favorable"),
        ("equal", 2.0, 1.0, "attention"),
    ],
)
def test_every_supported_comparison_operator(operator, value, threshold, expected_status) -> None:
    rule = approved_rule(operator=operator, threshold=threshold)

    result = evaluate_guidance({"commitment_change_pct": value}, {}, [rule])

    assert result["rule_results"][0]["status"] == expected_status


def test_evaluator_handles_baseline_comparison_and_report_only_rules() -> None:
    rules = [
        approved_rule(
            rule_id="SIMPLICITY-S-001",
            lens="simplicity_with_scale",
            measure="scenario_renewal_points",
            comparison_measure="baseline_renewal_points",
            operator="less_than_or_equal",
            explanation="Renewal decision points should not increase.",
        ),
        approved_rule(
            rule_id="CUSTOMER-CV-001",
            lens="customer_value",
            measure="flexible_adoption_pct",
            operator="report_only",
            threshold=None,
            status_if_matched="report_only",
            status_if_not_matched="report_only",
            explanation="Flexible adoption is informational.",
        ),
    ]

    result = evaluate_guidance(
        {"scenario_renewal_points": 12, "baseline_renewal_points": 10},
        {"flexible_adoption_pct": 75},
        rules,
    )

    by_rule = {row["rule_id"]: row for row in result["rule_results"]}
    assert by_rule["SIMPLICITY-S-001"]["evaluated_value"] == 2
    assert by_rule["SIMPLICITY-S-001"]["status"] == "attention"
    assert by_rule["CUSTOMER-CV-001"]["status"] == "report_only"


def test_missing_measure_produces_unknown_instead_of_crashing() -> None:
    result = evaluate_guidance({}, {}, [approved_rule()])

    assert result["lenses"]["growth"]["status"] == "unknown"
    assert "could not be evaluated" in result["rule_results"][0]["reason"]


def test_no_active_rules_produces_explicit_unavailable_lenses() -> None:
    result = evaluate_guidance({}, {}, [])

    assert result["applied"] is False
    assert result["applied_rule_count"] == 0
    assert all(lens["status"] == "not_available" for lens in result["lenses"].values())


def test_unapproved_rule_cannot_be_evaluated() -> None:
    rule = approved_rule(review_status="in_review")

    with pytest.raises(ValueError, match="Unapproved"):
        evaluate_guidance({"commitment_change_pct": 1}, {}, [rule])


def test_segment_rule_requires_an_exclusive_matching_filter() -> None:
    rule = approved_rule(
        rule_id="GROWTH-G-EMEA",
        scope_type="portfolio_filter",
        scope_column="SCMS",
        scope_value="ENTERPRISE",
    )

    matching = evaluate_guidance(
        {"commitment_change_pct": 1},
        {},
        [rule],
        {"SCMS": ["ENTERPRISE"]},
    )
    mixed = evaluate_guidance(
        {"commitment_change_pct": 1},
        {},
        [rule],
        {"SCMS": ["ENTERPRISE", "COMMERCIAL"]},
    )

    assert matching["applied_rule_count"] == 1
    assert mixed["applied_rule_count"] == 0


def _write_registry(path: Path, rule: GuidanceRule, active: bool = True) -> None:
    payload = {
        "schema_version": "1.0",
        "generated_at_utc": "2026-11-04T10:00:00+00:00",
        "candidate_rule_count": 1,
        "active_rule_count": 1 if active else 0,
        "active_rule_ids": [rule.rule_id] if active else [],
        "rules": [rule.to_dict()],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_registry_loader_revalidates_active_rules(tmp_path: Path) -> None:
    path = tmp_path / "guidance_rules.json"
    rule = approved_rule()
    _write_registry(path, rule)

    registry = load_guidance_registry(path)

    assert registry.active_rules == [rule]
    assert registry.generated_at_utc == "2026-11-04T10:00:00+00:00"


def test_registry_rejects_unapproved_active_rule(tmp_path: Path) -> None:
    path = tmp_path / "guidance_rules.json"
    _write_registry(path, approved_rule(review_status="in_review"))

    with pytest.raises(GuidanceRegistryError, match="approved review status"):
        load_guidance_registry(path)


def test_missing_guidance_registry_is_non_ambiguous(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_guidance_registry(tmp_path / "missing.json")


def test_user_facing_guidance_status_labels_are_stable() -> None:
    assert guidance_status_label("favorable") == "Favorable"
    assert guidance_status_label("attention") == "Attention required"
    assert guidance_status_label("not_available") == "No guidance rule applied"


@pytest.mark.parametrize(
    ("error", "candidate_count", "active_count", "applied_count", "expected"),
    [
        ("missing registry", 0, 0, 0, "error"),
        (None, 6, 0, 0, "unavailable"),
        (None, 6, 2, 0, "out_of_scope"),
        (None, 6, 2, 2, "applied"),
    ],
)
def test_streamlit_guidance_panel_states(
    error,
    candidate_count,
    active_count,
    applied_count,
    expected,
) -> None:
    state, message = guidance_panel_state(error, candidate_count, active_count, applied_count)

    assert state == expected
    assert message
