from datetime import datetime, timezone

import pytest

from scripts.validate_ea_guidance_ci import validate_ci_guidance


FIXED_EXTRACTION_TIME = datetime(2026, 11, 4, 12, 0, tzinfo=timezone.utc)


def test_ci_guidance_artifacts_are_deterministic_and_blocking_checks_pass() -> None:
    report = validate_ci_guidance("report", FIXED_EXTRACTION_TIME)

    assert report["failure_count"] == 0
    assert report["warning_count"] > 0
    assert report["candidate_rule_count"] == 6
    assert report["active_rule_count"] == 0
    assert set(report["artifact_sha256"]) == {
        "content",
        "quality",
        "rules",
        "compatibility",
        "summary",
        "candidate_review",
    }


def test_strict_warning_policy_blocks_current_in_review_fixture() -> None:
    with pytest.raises(ValueError, match="warnings are forbidden"):
        validate_ci_guidance("fail", FIXED_EXTRACTION_TIME)
