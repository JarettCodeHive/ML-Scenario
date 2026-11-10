from datetime import date

import pandas as pd
import pytest

from cluster_c.config import GUIDANCE_APPROVALS_PATH
from cluster_c.unstructured.approval import apply_human_approvals
from cluster_c.unstructured.preparation import prepare_guidance
from cluster_c.unstructured.rule_extractor import extract_candidate_rules


def _inputs():
    prepared = prepare_guidance()
    extracted = extract_candidate_rules(prepared.content)
    return prepared.content.copy(), extracted.rules


def test_sample_candidates_are_not_active() -> None:
    content, rules = _inputs()

    result = apply_human_approvals(rules, content, as_of=date(2026, 11, 4))

    assert result.is_valid
    assert len(result.candidate_rules) == 6
    assert result.active_rules == []
    assert all(rule.review_status == "in_review" for rule in result.candidate_rules)


def test_phase_five_review_gate_blocks_an_otherwise_approved_rule(tmp_path) -> None:
    content, rules = _inputs()
    content["review_status"] = "approved"
    approvals = pd.read_csv(GUIDANCE_APPROVALS_PATH, dtype="string", keep_default_na=False)
    approvals["review_status"] = "approved"
    approvals["reviewed_by"] = "ea-owner@example.test"
    approvals["reviewed_at_utc"] = "2026-11-04T10:00:00Z"
    approvals_path = tmp_path / "approvals.csv"
    approvals.to_csv(approvals_path, index=False)

    result = apply_human_approvals(
        rules,
        content,
        approvals_path=approvals_path,
        as_of=date(2026, 11, 4),
        candidate_reviewed_rule_keys=set(),
    )

    assert result.active_rules == []
    checks = result.checks.loc[result.checks.check_id == "candidate_review_required"]
    assert checks.status.eq("WARN").all()
    assert result.checks.loc[result.checks.check_id == "approved_source_required", "status"].eq("WARN").all()


def test_rule_becomes_active_only_with_complete_human_and_source_approval(tmp_path) -> None:
    content, rules = _inputs()
    content["review_status"] = "approved"
    approvals = pd.read_csv(GUIDANCE_APPROVALS_PATH, dtype="string", keep_default_na=False)
    approvals["review_status"] = "approved"
    approvals["reviewed_by"] = "ea-owner@example.test"
    approvals["reviewed_at_utc"] = "2026-11-04T10:00:00Z"
    approvals_path = tmp_path / "approvals.csv"
    approvals.to_csv(approvals_path, index=False)

    result = apply_human_approvals(
        rules,
        content,
        approvals_path=approvals_path,
        as_of=date(2026, 11, 4),
    )

    assert result.is_valid
    assert len(result.active_rules) == 6
    assert all(rule.review_status == "approved" for rule in result.active_rules)
    assert result.checks.loc[result.checks.check_id == "approved_source_required", "status"].eq("PASS").all()


def test_approved_rule_is_inactive_when_source_is_not_approved(tmp_path) -> None:
    content, rules = _inputs()
    approvals = pd.read_csv(GUIDANCE_APPROVALS_PATH, dtype="string", keep_default_na=False)
    approvals["review_status"] = "approved"
    approvals["reviewed_by"] = "ea-owner@example.test"
    approvals["reviewed_at_utc"] = "2026-11-04T10:00:00Z"
    approvals_path = tmp_path / "approvals.csv"
    approvals.to_csv(approvals_path, index=False)

    result = apply_human_approvals(rules, content, approvals_path, date(2026, 11, 4))

    assert result.is_valid
    assert result.active_rules == []


def test_approved_record_requires_reviewer_and_timestamp(tmp_path) -> None:
    content, rules = _inputs()
    approvals = pd.read_csv(GUIDANCE_APPROVALS_PATH, dtype="string", keep_default_na=False)
    approvals.loc[0, "review_status"] = "approved"
    approvals_path = tmp_path / "approvals.csv"
    approvals.to_csv(approvals_path, index=False)

    result = apply_human_approvals(rules, content, approvals_path, date(2026, 11, 4))

    assert not result.is_valid
    invalid = result.checks.loc[
        (result.checks.check_id == "approval_record_valid")
        & (result.checks.status == "FAIL")
    ]
    assert len(invalid) == 1
    assert "GROWTH-G-001" in invalid.iloc[0].detail
    assert result.active_rules == []


def test_approved_record_requires_authorized_reviewer(tmp_path) -> None:
    content, rules = _inputs()
    content["review_status"] = "approved"
    approvals = pd.read_csv(GUIDANCE_APPROVALS_PATH, dtype="string", keep_default_na=False)
    approvals["review_status"] = "approved"
    approvals["reviewed_by"] = "unauthorized@example.test"
    approvals["reviewed_at_utc"] = "2026-11-04T10:00:00Z"
    approvals_path = tmp_path / "approvals.csv"
    approvals.to_csv(approvals_path, index=False)

    result = apply_human_approvals(
        rules,
        content,
        approvals_path=approvals_path,
        as_of=date(2026, 11, 4),
        authorized_reviewers_by_source={"EA-GUIDE-SAMPLE-001": {"authorized@example.test"}},
    )

    assert not result.is_valid
    assert result.active_rules == []
    invalid = result.checks.loc[
        (result.checks.check_id == "approval_record_valid")
        & (result.checks.status == "FAIL")
    ]
    assert len(invalid) == len(rules)
    assert invalid.detail.str.contains("reviewer_authorized=False").all()


def test_hash_change_invalidates_prior_human_review() -> None:
    content, rules = _inputs()
    changed_rule = rules[0].__class__(**{
        **rules[0].to_dict(),
        "source_sha256": "0" * 64,
    })

    result = apply_human_approvals(
        [changed_rule],
        content,
        as_of=date(2026, 11, 4),
    )

    assert result.is_valid
    assert result.candidate_rules[0].review_status == "not_approved"
    assert result.active_rules == []


@pytest.mark.parametrize("source_status", ["retired", "rejected", "in_review"])
def test_nonapproved_source_lifecycle_status_never_activates_rules(tmp_path, source_status) -> None:
    content, rules = _inputs()
    content["review_status"] = source_status
    approvals = pd.read_csv(GUIDANCE_APPROVALS_PATH, dtype="string", keep_default_na=False)
    approvals["review_status"] = "approved"
    approvals["reviewed_by"] = "ea-owner@example.test"
    approvals["reviewed_at_utc"] = "2026-11-04T10:00:00Z"
    approvals_path = tmp_path / "approvals.csv"
    approvals.to_csv(approvals_path, index=False)

    result = apply_human_approvals(rules, content, approvals_path, date(2026, 11, 4))

    assert result.is_valid
    assert result.active_rules == []


def test_future_guidance_is_not_active_before_effective_date(tmp_path) -> None:
    content, rules = _inputs()
    content["review_status"] = "approved"
    content["effective_date"] = pd.Timestamp("2027-01-01")
    approvals = pd.read_csv(GUIDANCE_APPROVALS_PATH, dtype="string", keep_default_na=False)
    approvals["review_status"] = "approved"
    approvals["reviewed_by"] = "ea-owner@example.test"
    approvals["reviewed_at_utc"] = "2026-11-04T10:00:00Z"
    approvals_path = tmp_path / "approvals.csv"
    approvals.to_csv(approvals_path, index=False)

    result = apply_human_approvals(rules, content, approvals_path, date(2026, 11, 4))

    assert result.is_valid
    assert result.active_rules == []


def test_governance_gate_blocks_otherwise_approved_rules(tmp_path) -> None:
    content, rules = _inputs()
    content["review_status"] = "approved"
    approvals = pd.read_csv(GUIDANCE_APPROVALS_PATH, dtype="string", keep_default_na=False)
    approvals["review_status"] = "approved"
    approvals["reviewed_by"] = "ea-owner@example.test"
    approvals["reviewed_at_utc"] = "2026-11-04T10:00:00Z"
    approvals_path = tmp_path / "approvals.csv"
    approvals.to_csv(approvals_path, index=False)

    result = apply_human_approvals(
        rules,
        content,
        approvals_path,
        date(2026, 11, 4),
        governance_eligible_source_ids=set(),
    )

    assert result.is_valid
    assert result.active_rules == []
    governance_checks = result.checks.loc[result.checks.check_id == "source_governance_required"]
    assert governance_checks.status.eq("WARN").all()


def test_compatibility_gate_blocks_otherwise_approved_rules(tmp_path) -> None:
    content, rules = _inputs()
    content["review_status"] = "approved"
    approvals = pd.read_csv(GUIDANCE_APPROVALS_PATH, dtype="string", keep_default_na=False)
    approvals["review_status"] = "approved"
    approvals["reviewed_by"] = "ea-owner@example.test"
    approvals["reviewed_at_utc"] = "2026-11-04T10:00:00Z"
    approvals_path = tmp_path / "approvals.csv"
    approvals.to_csv(approvals_path, index=False)

    result = apply_human_approvals(
        rules,
        content,
        approvals_path,
        date(2026, 11, 4),
        governance_eligible_source_ids={"EA-GUIDE-SAMPLE-001"},
        compatibility_eligible_source_ids=set(),
    )

    assert result.is_valid
    assert result.active_rules == []
    checks = result.checks.loc[result.checks.check_id == "source_compatibility_required"]
    assert checks.status.eq("WARN").all()
