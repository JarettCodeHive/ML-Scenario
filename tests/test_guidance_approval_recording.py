from datetime import datetime, timezone

import pandas as pd
import pytest

from cluster_c.config import (
    GUIDANCE_APPROVALS_PATH,
    GUIDANCE_CANDIDATE_REVIEWS_PATH,
    GUIDANCE_GOVERNANCE_PATH,
    GUIDANCE_MANIFEST_PATH,
    GUIDANCE_REVIEW_POLICY_PATH,
    GUIDANCE_RULES_PATH,
    GUIDANCE_SOURCE_APPROVALS_PATH,
)
from cluster_c.unstructured.approval_recording import (
    record_rule_decision,
    record_source_decision,
    validate_source_approvals,
)
from cluster_c.unstructured.manifest import validate_guidance_manifest


SOURCE_ID = "EA-GUIDE-SAMPLE-001"
RULE_ID = "GROWTH-G-001"


def _copy_csv(source, destination):
    frame = pd.read_csv(source, dtype="string", keep_default_na=False)
    frame.to_csv(destination, index=False)
    return frame


def test_synthetic_rule_cannot_be_approved() -> None:
    with pytest.raises(ValueError, match="Synthetic guidance cannot be approved"):
        record_rule_decision(
            RULE_ID,
            decision="approved",
            reviewed_by="anyone",
            review_notes="Must remain blocked",
            dry_run=True,
        )


def test_rejected_rule_decision_is_recorded_with_evidence(tmp_path) -> None:
    approvals_path = tmp_path / "rule_approvals.csv"
    candidate_reviews_path = tmp_path / "candidate_reviews.csv"
    governance_path = tmp_path / "governance.csv"
    _copy_csv(GUIDANCE_APPROVALS_PATH, approvals_path)
    reviews = _copy_csv(GUIDANCE_CANDIDATE_REVIEWS_PATH, candidate_reviews_path)
    governance = _copy_csv(GUIDANCE_GOVERNANCE_PATH, governance_path)
    reviews.loc[reviews.rule_id == RULE_ID, [
        "review_decision",
        "page_compared",
        "ambiguity_resolved",
        "measure_operator_confirmed",
        "scope_confirmed",
        "reviewed_by",
        "reviewed_at_utc",
        "review_notes",
    ]] = [
        "rejected",
        "true",
        "false",
        "false",
        "false",
        "phase5-reviewer",
        datetime.now(timezone.utc).isoformat(),
        "Source language is ambiguous",
    ]
    reviews.to_csv(candidate_reviews_path, index=False)
    governance.loc[governance.source_id == SOURCE_ID, "authorized_rule_reviewer"] = "rule-reviewer"
    governance.to_csv(governance_path, index=False)

    record_rule_decision(
        RULE_ID,
        decision="rejected",
        reviewed_by="rule-reviewer",
        review_notes="Rejected after page comparison",
        approvals_path=approvals_path,
        candidate_reviews_path=candidate_reviews_path,
        review_policy_path=GUIDANCE_REVIEW_POLICY_PATH,
        governance_path=governance_path,
        manifest_path=GUIDANCE_MANIFEST_PATH,
        rules_path=GUIDANCE_RULES_PATH,
    )

    restored = pd.read_csv(approvals_path, dtype="string", keep_default_na=False)
    row = restored.loc[restored.rule_id == RULE_ID].iloc[0]
    assert row.review_status == "rejected"
    assert row.reviewed_by == "rule-reviewer"
    assert row.review_notes == "Rejected after page comparison"


def test_rule_decision_rejects_unauthorized_reviewer(tmp_path) -> None:
    governance_path = tmp_path / "governance.csv"
    _copy_csv(GUIDANCE_GOVERNANCE_PATH, governance_path)

    with pytest.raises(ValueError, match="not authorized"):
        record_rule_decision(
            RULE_ID,
            decision="rejected",
            reviewed_by="not-authorized",
            review_notes="No authority",
            governance_path=governance_path,
            dry_run=True,
        )


def test_source_rejection_atomically_updates_manifest_and_evidence(tmp_path) -> None:
    manifest_path = tmp_path / "sources.csv"
    governance_path = tmp_path / "governance.csv"
    source_approvals_path = tmp_path / "source_approvals.csv"
    _copy_csv(GUIDANCE_MANIFEST_PATH, manifest_path)
    _copy_csv(GUIDANCE_GOVERNANCE_PATH, governance_path)
    _copy_csv(GUIDANCE_SOURCE_APPROVALS_PATH, source_approvals_path)

    record_source_decision(
        SOURCE_ID,
        decision="rejected",
        reviewed_by="POC_TEST_FIXTURE",
        review_notes="Synthetic fixture remains excluded",
        manifest_path=manifest_path,
        governance_path=governance_path,
        source_approvals_path=source_approvals_path,
    )

    manifest = pd.read_csv(manifest_path, dtype="string", keep_default_na=False)
    evidence = pd.read_csv(source_approvals_path, dtype="string", keep_default_na=False)
    assert manifest.loc[manifest.source_id == SOURCE_ID, "review_status"].iloc[0] == "rejected"
    row = evidence.loc[evidence.source_id == SOURCE_ID].iloc[0]
    assert row.review_status == "rejected"
    assert row.reviewed_by == "POC_TEST_FIXTURE"
    assert row.review_notes == "Synthetic fixture remains excluded"


def test_source_approval_validator_requires_authorized_hash_bound_evidence(tmp_path) -> None:
    manifest = validate_guidance_manifest().manifest.copy()
    manifest.loc[:, "is_synthetic"] = False
    manifest.loc[:, "review_status"] = "approved"
    governance = pd.read_csv(GUIDANCE_GOVERNANCE_PATH, dtype="string", keep_default_na=False)
    approvals = pd.read_csv(GUIDANCE_SOURCE_APPROVALS_PATH, dtype="string", keep_default_na=False)
    approvals.loc[:, "review_status"] = "approved"
    approvals.loc[:, "reviewed_by"] = "POC_TEST_FIXTURE"
    approvals.loc[:, "reviewed_at_utc"] = datetime.now(timezone.utc).isoformat()
    approvals.loc[:, "review_notes"] = "Document-level review completed"
    approvals_path = tmp_path / "source_approvals.csv"
    approvals.to_csv(approvals_path, index=False)

    result = validate_source_approvals(manifest, governance, approvals_path)

    assert result.is_valid
    assert result.eligible_source_ids == {SOURCE_ID}


def test_source_approval_validator_blocks_manual_synthetic_approval(tmp_path) -> None:
    manifest = validate_guidance_manifest().manifest.copy()
    manifest.loc[:, "review_status"] = "approved"
    governance = pd.read_csv(GUIDANCE_GOVERNANCE_PATH, dtype="string", keep_default_na=False)
    approvals = pd.read_csv(GUIDANCE_SOURCE_APPROVALS_PATH, dtype="string", keep_default_na=False)
    approvals.loc[:, "review_status"] = "approved"
    approvals.loc[:, "reviewed_by"] = "POC_TEST_FIXTURE"
    approvals.loc[:, "reviewed_at_utc"] = datetime.now(timezone.utc).isoformat()
    approvals.loc[:, "review_notes"] = "Attempted manual approval"
    approvals_path = tmp_path / "source_approvals.csv"
    approvals.to_csv(approvals_path, index=False)

    result = validate_source_approvals(manifest, governance, approvals_path)

    assert not result.is_valid
    assert result.eligible_source_ids == set()
    failure = result.checks.loc[result.checks.check_id == "source_approval_record_valid"].iloc[0]
    assert "synthetic_approved=True" in failure.detail
