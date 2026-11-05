from datetime import datetime, timezone

import pandas as pd

from cluster_c.config import (
    GUIDANCE_CANDIDATE_REVIEWS_PATH,
    GUIDANCE_REVIEW_POLICY_PATH,
)
from cluster_c.unstructured.candidate_review import (
    CANDIDATE_REVIEW_COLUMNS,
    EXPECTED_STATUS_PRECEDENCE,
    candidate_sha256,
    initialize_candidate_reviews,
    validate_candidate_reviews,
)
from cluster_c.unstructured.preparation import prepare_guidance


def _rules():
    return prepare_guidance().candidate_rules


def _confirmed_registries(tmp_path, rules, rejected_rule_id=None):
    reviewed_at = datetime(2026, 11, 4, 12, 0, tzinfo=timezone.utc).isoformat()
    reviews = []
    for rule in rules:
        rejected = rule.rule_id == rejected_rule_id
        reviews.append({
            "rule_id": rule.rule_id,
            "source_id": rule.source_id,
            "source_version": rule.source_version,
            "source_sha256": rule.source_sha256,
            "candidate_sha256": candidate_sha256(rule),
            "review_decision": "rejected" if rejected else "confirmed",
            "page_compared": True,
            "ambiguity_resolved": not rejected,
            "measure_operator_confirmed": not rejected,
            "scope_confirmed": not rejected,
            "reviewed_by": "ea-owner@example.test",
            "reviewed_at_utc": reviewed_at,
            "review_notes": "Compared with the cited source page",
        })
    reviews_path = tmp_path / "candidate_reviews.csv"
    pd.DataFrame(reviews, columns=CANDIDATE_REVIEW_COLUMNS).to_csv(reviews_path, index=False)
    policy_path = tmp_path / "review_policy.csv"
    pd.DataFrame([{
        "policy_id": "lens_status_precedence",
        "status_precedence": EXPECTED_STATUS_PRECEDENCE,
        "review_status": "confirmed",
        "reviewed_by": "ea-owner@example.test",
        "reviewed_at_utc": reviewed_at,
        "review_notes": "Conservative precedence accepted",
    }]).to_csv(policy_path, index=False)
    return reviews_path, policy_path


def test_sample_candidate_reviews_remain_incomplete_and_inactive() -> None:
    result = validate_candidate_reviews(
        _rules(),
        GUIDANCE_CANDIDATE_REVIEWS_PATH,
        GUIDANCE_REVIEW_POLICY_PATH,
    )

    assert result.is_valid
    assert not result.is_complete
    assert not result.precedence_confirmed
    assert result.confirmed_rule_keys == set()
    assert result.report["review_decision"].eq("in_review").all()


def test_confirmed_hash_bound_reviews_are_activation_eligible(tmp_path) -> None:
    rules = _rules()
    reviews_path, policy_path = _confirmed_registries(tmp_path, rules)

    result = validate_candidate_reviews(rules, reviews_path, policy_path)

    assert result.is_valid
    assert result.is_complete
    assert result.precedence_confirmed
    assert len(result.confirmed_rule_keys) == len(rules)
    assert result.report["activation_review_eligible"].all()


def test_rejected_candidate_is_complete_but_never_activation_eligible(tmp_path) -> None:
    rules = _rules()
    rejected_rule_id = rules[0].rule_id
    reviews_path, policy_path = _confirmed_registries(tmp_path, rules, rejected_rule_id)

    result = validate_candidate_reviews(rules, reviews_path, policy_path)

    rejected = result.report.loc[result.report.rule_id == rejected_rule_id].iloc[0]
    assert result.is_complete
    assert bool(rejected.review_complete)
    assert not bool(rejected.activation_review_eligible)
    assert len(result.confirmed_rule_keys) == len(rules) - 1


def test_changed_candidate_fingerprint_invalidates_previous_review(tmp_path) -> None:
    rules = _rules()
    reviews_path, policy_path = _confirmed_registries(tmp_path, rules)
    reviews = pd.read_csv(reviews_path, dtype="string", keep_default_na=False)
    reviews.loc[0, "candidate_sha256"] = "0" * 64
    reviews.to_csv(reviews_path, index=False)

    result = validate_candidate_reviews(rules, reviews_path, policy_path)

    assert not result.is_valid
    failure = result.checks.loc[result.checks.check_id == "candidate_review_record_valid"].iloc[0]
    assert rules[0].rule_id in failure.detail


def test_initialize_appends_missing_rows_without_replacing_existing_review(tmp_path) -> None:
    rules = _rules()
    reviews_path, _ = _confirmed_registries(tmp_path, rules[:1])

    added = initialize_candidate_reviews(rules, reviews_path)
    restored = pd.read_csv(reviews_path, dtype="string", keep_default_na=False)

    assert added == len(rules) - 1
    assert len(restored) == len(rules)
    assert restored.loc[restored.rule_id == rules[0].rule_id, "review_decision"].iloc[0] == "confirmed"
