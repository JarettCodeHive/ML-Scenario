"""Guarded recording and validation of Phase 6 EA guidance approvals."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from cluster_c.config import (
    GUIDANCE_APPROVALS_PATH,
    GUIDANCE_CANDIDATE_REVIEWS_PATH,
    GUIDANCE_COMPATIBILITY_PATH,
    GUIDANCE_GOVERNANCE_PATH,
    GUIDANCE_MANIFEST_PATH,
    GUIDANCE_PDF_DIR,
    GUIDANCE_REVIEW_POLICY_PATH,
    GUIDANCE_RULES_PATH,
    GUIDANCE_SOURCE_APPROVALS_PATH,
    PROJECT_ROOT,
)

from .approval import APPROVAL_COLUMNS
from .candidate_review import validate_candidate_reviews
from .compatibility import assess_guidance_compatibility
from .decision_rules import load_guidance_registry
from .governance import GOVERNANCE_COLUMNS, validate_guidance_governance
from .manifest import (
    ALLOWED_REVIEW_STATUSES,
    QUALITY_COLUMNS,
    REQUIRED_MANIFEST_COLUMNS,
    validate_guidance_manifest,
)


SOURCE_APPROVAL_COLUMNS = [
    "source_id",
    "source_version",
    "source_sha256",
    "review_status",
    "reviewed_by",
    "reviewed_at_utc",
    "review_notes",
]


@dataclass(frozen=True)
class SourceApprovalValidationResult:
    approvals: pd.DataFrame
    checks: pd.DataFrame
    eligible_source_ids: set[str]

    @property
    def is_valid(self) -> bool:
        return not self.checks["status"].eq("FAIL").any()


@dataclass(frozen=True)
class ApprovalRecordingResult:
    subject_id: str
    decision: str
    reviewer: str
    reviewed_at_utc: str
    dry_run: bool


def _check(check_id: str, source_id: str, status: str, severity: str, description: str, detail: str = "") -> dict[str, str]:
    return {
        "check_id": check_id,
        "source_id": source_id,
        "status": status,
        "severity": severity,
        "description": description,
        "detail": detail,
    }


def _read_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype="string", keep_default_na=False)
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Registry {path} is missing columns: {missing}")
    return frame


def _review_timestamp(value: datetime | None) -> datetime:
    reviewed_at = value or datetime.now(timezone.utc)
    if reviewed_at.tzinfo is None or reviewed_at.utcoffset() is None:
        raise ValueError("reviewed_at must include a UTC offset")
    reviewed_at = reviewed_at.astimezone(timezone.utc)
    if reviewed_at > datetime.now(timezone.utc):
        raise ValueError("reviewed_at cannot be in the future")
    return reviewed_at


def _is_true(value: object) -> bool:
    return value is True or str(value).strip().casefold() in {"true", "1", "yes"}


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _governance_reviewers(governance: pd.DataFrame, source_id: str, *, source_approval: bool) -> set[str]:
    rows = governance.loc[governance.source_id == source_id]
    if len(rows) != 1:
        raise ValueError(f"Exactly one governance record is required for {source_id}")
    row = rows.iloc[0]
    columns = (
        ("business_owner", "governance_contact")
        if source_approval
        else ("authorized_rule_reviewer", "backup_reviewer")
    )
    return {str(row[column]).strip() for column in columns if str(row[column]).strip()}


def validate_source_approvals(
    manifest: pd.DataFrame,
    governance: pd.DataFrame,
    approvals_path: Path = GUIDANCE_SOURCE_APPROVALS_PATH,
) -> SourceApprovalValidationResult:
    """Require hash-bound source approval evidence for approved manifests."""
    checks: list[dict[str, str]] = []
    try:
        approvals = _read_csv(approvals_path, SOURCE_APPROVAL_COLUMNS)
    except (OSError, UnicodeError, pd.errors.ParserError, ValueError) as exc:
        checks.append(_check("source_approval_registry_valid", "<SOURCE_APPROVALS>", "FAIL", "blocking", "Source approval registry must be readable and complete", str(exc)))
        return SourceApprovalValidationResult(
            pd.DataFrame(columns=SOURCE_APPROVAL_COLUMNS),
            pd.DataFrame(checks, columns=QUALITY_COLUMNS),
            set(),
        )
    key_columns = ["source_id", "source_version", "source_sha256"]
    duplicate_mask = approvals.duplicated(key_columns, keep=False)
    checks.append(_check(
        "source_approval_key_unique",
        "<SOURCE_APPROVALS>",
        "FAIL" if duplicate_mask.any() else "PASS",
        "blocking",
        "Each hash-bound source may have only one approval record",
        f"duplicates={int(duplicate_mask.sum())}" if duplicate_mask.any() else "",
    ))
    by_key = {
        tuple(str(row[column]) for column in key_columns): row
        for _, row in approvals.iterrows()
    }
    eligible: set[str] = set()
    manifest_keys = set()
    for source in manifest.itertuples(index=False):
        key = (str(source.source_id), str(source.version), str(source.sha256))
        manifest_keys.add(key)
        row = by_key.get(key)
        approved_manifest = source.review_status == "approved"
        synthetic_approved = approved_manifest and _is_true(source.is_synthetic)
        record_present = row is not None
        if not record_present:
            checks.append(_check(
                "source_approval_record_present",
                source.source_id,
                "FAIL" if approved_manifest else "WARN",
                "blocking" if approved_manifest else "review_required",
                "Every source requires hash-bound Phase 6 approval evidence before activation",
            ))
            continue
        reviewed_at = pd.to_datetime(row["reviewed_at_utc"], errors="coerce", utc=True)
        status_allowed = row["review_status"] in ALLOWED_REVIEW_STATUSES
        evidence = (
            row["review_status"] not in {"approved", "rejected"}
            or (
                bool(str(row["reviewed_by"]).strip())
                and not pd.isna(reviewed_at)
                and bool(str(row["review_notes"]).strip())
            )
        )
        try:
            authorized = (
                row["review_status"] not in {"approved", "rejected"}
                or row["reviewed_by"] in _governance_reviewers(governance, source.source_id, source_approval=True)
            )
        except ValueError:
            authorized = False
        status_matches = row["review_status"] == source.review_status
        valid = status_allowed and evidence and authorized and status_matches and not synthetic_approved
        checks.append(_check(
            "source_approval_record_valid",
            source.source_id,
            "PASS" if valid else ("FAIL" if approved_manifest else "WARN"),
            "blocking" if approved_manifest else "review_required",
            "Source status requires matching evidence from an authorized source approver",
            f"manifest_status={source.review_status}; approval_status={row['review_status']}; "
            f"reviewer_authorized={authorized}; synthetic_approved={synthetic_approved}",
        ))
        if valid and approved_manifest:
            eligible.add(str(source.source_id))
    for key in sorted(set(by_key) - manifest_keys):
        checks.append(_check("source_approval_matches_source", key[0], "WARN", "review_required", "Source approval must match a current manifest version and hash"))
    return SourceApprovalValidationResult(
        approvals,
        pd.DataFrame(checks, columns=QUALITY_COLUMNS),
        eligible,
    )


def record_rule_decision(
    rule_id: str,
    *,
    decision: str,
    reviewed_by: str,
    review_notes: str,
    reviewed_at: datetime | None = None,
    dry_run: bool = False,
    rules_path: Path = GUIDANCE_RULES_PATH,
    approvals_path: Path = GUIDANCE_APPROVALS_PATH,
    candidate_reviews_path: Path = GUIDANCE_CANDIDATE_REVIEWS_PATH,
    review_policy_path: Path = GUIDANCE_REVIEW_POLICY_PATH,
    governance_path: Path = GUIDANCE_GOVERNANCE_PATH,
    manifest_path: Path = GUIDANCE_MANIFEST_PATH,
) -> ApprovalRecordingResult:
    """Record an approved or rejected rule decision after Phase 5 review."""
    if decision not in {"approved", "rejected"}:
        raise ValueError("Rule decision must be 'approved' or 'rejected'")
    if not reviewed_by.strip() or not review_notes.strip():
        raise ValueError("Reviewer identity and review rationale are required")
    reviewed_at = _review_timestamp(reviewed_at)
    registry = load_guidance_registry(rules_path)
    matches = [rule for rule in registry.candidate_rules if rule.rule_id == rule_id]
    if len(matches) != 1:
        raise ValueError(f"Exactly one extracted candidate is required for rule_id={rule_id!r}")
    rule = matches[0]
    manifest = _read_csv(manifest_path, REQUIRED_MANIFEST_COLUMNS)
    source_rows = manifest.loc[manifest.source_id == rule.source_id]
    if len(source_rows) != 1:
        raise ValueError(f"Exactly one source manifest record is required for {rule.source_id}")
    if decision == "approved" and _is_true(source_rows.iloc[0]["is_synthetic"]):
        raise ValueError("Synthetic guidance cannot be approved for decision use")
    governance = _read_csv(governance_path, GOVERNANCE_COLUMNS)
    authorized = _governance_reviewers(governance, rule.source_id, source_approval=False)
    if reviewed_by not in authorized:
        raise ValueError(f"Reviewer {reviewed_by!r} is not authorized for source {rule.source_id}")
    phase5 = validate_candidate_reviews(
        registry.candidate_rules,
        reviews_path=candidate_reviews_path,
        policy_path=review_policy_path,
    )
    report = phase5.report.loc[phase5.report.rule_id == rule_id]
    if len(report) != 1:
        raise ValueError(f"Phase 5 review report is missing for {rule_id}")
    phase5_row = report.iloc[0]
    expected_phase5_decision = "confirmed" if decision == "approved" else "rejected"
    if (
        not phase5.is_valid
        or not bool(phase5_row["review_complete"])
        or phase5_row["review_decision"] != expected_phase5_decision
        or (decision == "approved" and not bool(phase5_row["activation_review_eligible"]))
    ):
        raise ValueError(f"Rule {rule_id} has not passed the required Phase 5 decision")

    approvals = _read_csv(approvals_path, APPROVAL_COLUMNS)
    key_columns = ["rule_id", "source_id", "source_version", "source_sha256"]
    key = [rule.rule_id, rule.source_id, rule.source_version, rule.source_sha256]
    mask = pd.Series(True, index=approvals.index)
    for column, value in zip(key_columns, key):
        mask &= approvals[column].eq(value)
    if mask.sum() > 1:
        raise ValueError(f"Duplicate rule approval records exist for {rule_id}")
    record = dict(zip(key_columns, key))
    record.update({
        "review_status": decision,
        "reviewed_by": reviewed_by,
        "reviewed_at_utc": reviewed_at.isoformat(),
        "review_notes": review_notes.strip(),
    })
    if mask.any():
        for column, value in record.items():
            approvals.loc[mask, column] = value
    else:
        approvals = pd.concat([approvals, pd.DataFrame([record])], ignore_index=True)
    if not dry_run:
        _atomic_write_csv(approvals[APPROVAL_COLUMNS], approvals_path)
    return ApprovalRecordingResult(rule_id, decision, reviewed_by, reviewed_at.isoformat(), dry_run)


def record_source_decision(
    source_id: str,
    *,
    decision: str,
    reviewed_by: str,
    review_notes: str,
    reviewed_at: datetime | None = None,
    dry_run: bool = False,
    project_root: Path = PROJECT_ROOT,
    guidance_pdf_dir: Path = GUIDANCE_PDF_DIR,
    manifest_path: Path = GUIDANCE_MANIFEST_PATH,
    governance_path: Path = GUIDANCE_GOVERNANCE_PATH,
    compatibility_path: Path = GUIDANCE_COMPATIBILITY_PATH,
    source_approvals_path: Path = GUIDANCE_SOURCE_APPROVALS_PATH,
    rule_approvals_path: Path = GUIDANCE_APPROVALS_PATH,
    candidate_reviews_path: Path = GUIDANCE_CANDIDATE_REVIEWS_PATH,
    review_policy_path: Path = GUIDANCE_REVIEW_POLICY_PATH,
    rules_path: Path = GUIDANCE_RULES_PATH,
) -> ApprovalRecordingResult:
    """Atomically record source evidence and update its manifest status."""
    if decision not in {"approved", "rejected"}:
        raise ValueError("Source decision must be 'approved' or 'rejected'")
    if not reviewed_by.strip() or not review_notes.strip():
        raise ValueError("Reviewer identity and review rationale are required")
    reviewed_at = _review_timestamp(reviewed_at)
    manifest_result = validate_guidance_manifest(manifest_path, project_root, guidance_pdf_dir)
    if not manifest_result.is_valid:
        raise ValueError("Guidance source manifest must be valid before recording approval")
    manifest = _read_csv(manifest_path, REQUIRED_MANIFEST_COLUMNS)
    mask = manifest.source_id.eq(source_id)
    if mask.sum() != 1:
        raise ValueError(f"Exactly one source manifest record is required for {source_id}")
    source = manifest.loc[mask].iloc[0]
    if decision == "approved" and _is_true(source["is_synthetic"]):
        raise ValueError("Synthetic guidance cannot be approved for decision use")
    governance = _read_csv(governance_path, GOVERNANCE_COLUMNS)
    authorized = _governance_reviewers(governance, source_id, source_approval=True)
    if reviewed_by not in authorized:
        raise ValueError(f"Reviewer {reviewed_by!r} is not authorized to approve source {source_id}")

    if decision == "approved":
        proposed = manifest_result.manifest.copy()
        proposed.loc[proposed.source_id == source_id, "review_status"] = "approved"
        governance_result = validate_guidance_governance(proposed, governance_path, reviewed_at.date())
        compatibility_result = assess_guidance_compatibility(proposed, compatibility_path, reviewed_at.date())
        if source_id not in governance_result.eligible_source_ids:
            raise ValueError(f"Source governance is not activation-ready for {source_id}")
        if source_id not in compatibility_result.eligible_source_ids:
            raise ValueError(f"Source compatibility is not activation-ready for {source_id}")
        registry = load_guidance_registry(rules_path)
        source_rules = [rule for rule in registry.candidate_rules if rule.source_id == source_id]
        phase5 = validate_candidate_reviews(source_rules, candidate_reviews_path, review_policy_path)
        if not source_rules or not phase5.is_complete:
            raise ValueError(f"Phase 5 review is incomplete for source {source_id}")
        rule_approvals = _read_csv(rule_approvals_path, APPROVAL_COLUMNS)
        decisions = {
            (row.rule_id, row.source_id, row.source_version, row.source_sha256): row.review_status
            for row in rule_approvals.itertuples(index=False)
        }
        incomplete = [
            rule.rule_id
            for rule in source_rules
            if decisions.get((rule.rule_id, rule.source_id, rule.source_version, rule.source_sha256))
            not in {"approved", "rejected"}
        ]
        if incomplete:
            raise ValueError(f"Phase 6 rule decisions are incomplete: {incomplete}")

    source_approvals = _read_csv(source_approvals_path, SOURCE_APPROVAL_COLUMNS)
    source_key_columns = ["source_id", "source_version", "source_sha256"]
    source_key = [source_id, str(source["version"]), str(source["sha256"])]
    approval_mask = pd.Series(True, index=source_approvals.index)
    for column, value in zip(source_key_columns, source_key):
        approval_mask &= source_approvals[column].eq(value)
    if approval_mask.sum() > 1:
        raise ValueError(f"Duplicate source approval records exist for {source_id}")
    record = dict(zip(source_key_columns, source_key))
    record.update({
        "review_status": decision,
        "reviewed_by": reviewed_by,
        "reviewed_at_utc": reviewed_at.isoformat(),
        "review_notes": review_notes.strip(),
    })
    if approval_mask.any():
        for column, value in record.items():
            source_approvals.loc[approval_mask, column] = value
    else:
        source_approvals = pd.concat([source_approvals, pd.DataFrame([record])], ignore_index=True)
    manifest.loc[mask, "review_status"] = decision
    if not dry_run:
        originals = {
            manifest_path: manifest_path.read_bytes(),
            source_approvals_path: source_approvals_path.read_bytes(),
        }
        try:
            _atomic_write_csv(source_approvals[SOURCE_APPROVAL_COLUMNS], source_approvals_path)
            _atomic_write_csv(manifest, manifest_path)
        except Exception:
            for path, content in originals.items():
                path.write_bytes(content)
            raise
    return ApprovalRecordingResult(source_id, decision, reviewed_by, reviewed_at.isoformat(), dry_run)
