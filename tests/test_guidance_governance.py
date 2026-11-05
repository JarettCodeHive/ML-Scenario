from datetime import date
from pathlib import Path

import pandas as pd

from cluster_c.config import GUIDANCE_GOVERNANCE_PATH
from cluster_c.unstructured.governance import validate_guidance_governance
from cluster_c.unstructured.manifest import validate_guidance_manifest


def test_synthetic_source_is_extractable_but_not_governance_eligible() -> None:
    manifest = validate_guidance_manifest().manifest

    result = validate_guidance_governance(manifest, as_of=date(2026, 11, 4))

    assert result.is_valid
    assert result.eligible_source_ids == set()
    assert result.checks.status.eq("FAIL").sum() == 0
    assert "governance_activation_eligible" in set(result.checks.loc[result.checks.status == "WARN", "check_id"])


def approved_governance_frame() -> pd.DataFrame:
    governance = pd.read_csv(GUIDANCE_GOVERNANCE_PATH, dtype="string", keep_default_na=False)
    governance.loc[0, [
        "business_owner",
        "technical_owner",
        "authorized_rule_reviewer",
        "backup_reviewer",
        "source_repository",
        "governance_contact",
    ]] = [
        "EA Business Owner",
        "EA Model Owner",
        "EA Authorized Reviewer",
        "EA Backup Reviewer",
        "Approved Document Repository",
        "ea-governance@example.test",
    ]
    governance.loc[0, "authority_status"] = "approved"
    governance.loc[0, "authority_confirmed_by"] = "EA Governance Board"
    governance.loc[0, "authority_confirmed_at_utc"] = "2026-11-01T10:00:00Z"
    governance.loc[0, "retention_policy"] = "Retain according to approved EA policy"
    governance.loc[0, "access_requirements"] = "Authorized EA users only"
    governance.loc[0, "review_frequency_days"] = "365"
    return governance


def test_complete_approved_governance_makes_source_eligible(tmp_path: Path) -> None:
    manifest = validate_guidance_manifest().manifest.copy()
    manifest.loc[0, "review_status"] = "approved"
    governance_path = tmp_path / "guidance_governance.csv"
    approved_governance_frame().to_csv(governance_path, index=False)

    result = validate_guidance_governance(
        manifest,
        governance_path=governance_path,
        as_of=date(2026, 11, 4),
    )

    assert result.is_valid
    assert result.eligible_source_ids == {"EA-GUIDE-SAMPLE-001"}
    eligible = result.checks.loc[result.checks.check_id == "governance_activation_eligible"].iloc[0]
    assert eligible.status == "PASS"


def test_approved_source_with_missing_owner_is_blocking(tmp_path: Path) -> None:
    manifest = validate_guidance_manifest().manifest.copy()
    manifest.loc[0, "review_status"] = "approved"
    governance = approved_governance_frame()
    governance.loc[0, "business_owner"] = ""
    governance_path = tmp_path / "guidance_governance.csv"
    governance.to_csv(governance_path, index=False)

    result = validate_guidance_governance(
        manifest,
        governance_path=governance_path,
        as_of=date(2026, 11, 4),
    )

    assert not result.is_valid
    failure = result.checks.loc[result.checks.check_id == "governance_owners_complete"].iloc[0]
    assert failure.status == "FAIL"
    assert "business_owner" in failure.detail
