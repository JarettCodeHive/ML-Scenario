"""Human-review controls for extracted EA guidance rule candidates."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Mapping

import pandas as pd

from cluster_c.config import GUIDANCE_APPROVALS_PATH

from .manifest import ALLOWED_REVIEW_STATUSES, QUALITY_COLUMNS, SHA256_PATTERN
from .schemas import GuidanceRule


APPROVAL_COLUMNS = [
    "rule_id",
    "source_id",
    "source_version",
    "source_sha256",
    "review_status",
    "reviewed_by",
    "reviewed_at_utc",
    "review_notes",
]


@dataclass(frozen=True)
class ApprovalResult:
    candidate_rules: list[GuidanceRule]
    active_rules: list[GuidanceRule]
    checks: pd.DataFrame

    @property
    def is_valid(self) -> bool:
        return not self.checks["status"].eq("FAIL").any()


def _check(
    check_id: str,
    source_id: str,
    status: str,
    severity: str,
    description: str,
    detail: str = "",
) -> dict[str, str]:
    return {
        "check_id": check_id,
        "source_id": source_id,
        "status": status,
        "severity": severity,
        "description": description,
        "detail": detail,
    }


def apply_human_approvals(
    rules: list[GuidanceRule],
    content: pd.DataFrame,
    approvals_path: Path = GUIDANCE_APPROVALS_PATH,
    as_of: date | None = None,
    governance_eligible_source_ids: set[str] | None = None,
    compatibility_eligible_source_ids: set[str] | None = None,
    candidate_reviewed_rule_keys: set[tuple[str, str, str, str]] | None = None,
    authorized_reviewers_by_source: Mapping[str, set[str]] | None = None,
    source_approval_eligible_source_ids: set[str] | None = None,
) -> ApprovalResult:
    """Apply hash-bound reviews and return only production-eligible rules."""
    checks: list[dict[str, str]] = []
    as_of = as_of or datetime.now(timezone.utc).date()
    try:
        approvals = pd.read_csv(approvals_path, dtype="string", keep_default_na=False)
    except (OSError, pd.errors.ParserError, UnicodeError) as exc:
        checks.append(_check("approval_registry_readable", "<APPROVALS>", "FAIL", "blocking", "Approval registry must be a readable CSV", str(exc)))
        return ApprovalResult(rules, [], pd.DataFrame(checks, columns=QUALITY_COLUMNS))

    missing = sorted(set(APPROVAL_COLUMNS) - set(approvals.columns))
    checks.append(_check(
        "approval_registry_columns",
        "<APPROVALS>",
        "PASS" if not missing else "FAIL",
        "blocking",
        "Approval registry must contain all required columns",
        f"missing={missing}" if missing else "",
    ))
    if missing:
        return ApprovalResult(rules, [], pd.DataFrame(checks, columns=QUALITY_COLUMNS))

    key_columns = ["rule_id", "source_id", "source_version", "source_sha256"]
    duplicate_mask = approvals.duplicated(key_columns, keep=False)
    duplicate_keys = (
        approvals.loc[duplicate_mask, key_columns]
        .astype(str)
        .apply(lambda row: "|".join(row), axis=1)
        .tolist()
        if duplicate_mask.any()
        else []
    )
    checks.append(_check(
        "approval_key_unique",
        "<APPROVALS>",
        "PASS" if not duplicate_keys else "FAIL",
        "blocking",
        "Each hash-bound candidate may have only one approval record",
        f"duplicates={duplicate_keys}" if duplicate_keys else "",
    ))

    valid_rows: dict[tuple[str, str, str, str], pd.Series] = {}
    for row in approvals.itertuples(index=False):
        key = (row.rule_id, row.source_id, row.source_version, row.source_sha256)
        status_allowed = row.review_status in ALLOWED_REVIEW_STATUSES
        hash_valid = bool(SHA256_PATTERN.fullmatch(row.source_sha256))
        reviewed_at = pd.to_datetime(row.reviewed_at_utc, errors="coerce") if row.reviewed_at_utc else pd.NaT
        final_decision = row.review_status in {"approved", "rejected"}
        review_evidence_complete = (
            not final_decision
            or (
                bool(row.reviewed_by.strip())
                and not pd.isna(reviewed_at)
                and bool(row.review_notes.strip())
            )
        )
        reviewer_authorized = (
            not final_decision
            or authorized_reviewers_by_source is None
            or row.reviewed_by in authorized_reviewers_by_source.get(row.source_id, set())
        )
        row_valid = (
            status_allowed
            and hash_valid
            and review_evidence_complete
            and reviewer_authorized
            and key not in valid_rows
        )
        checks.append(_check(
            "approval_record_valid",
            row.source_id,
            "PASS" if row_valid else "FAIL",
            "blocking",
            "Approval records require controlled status, valid hash, and reviewer evidence when approved",
            f"rule={row.rule_id}; status={row.review_status}; "
            f"reviewer_authorized={reviewer_authorized}",
        ))
        if row_valid:
            valid_rows[key] = pd.Series(row._asdict())

    source_rows = (
        content.sort_values("page_number")
        .drop_duplicates(["source_id", "source_version", "source_sha256"])
        .set_index(["source_id", "source_version", "source_sha256"])
        if not content.empty
        else pd.DataFrame()
    )
    reviewed_rules: list[GuidanceRule] = []
    active_rules: list[GuidanceRule] = []
    matched_keys: set[tuple[str, str, str, str]] = set()
    for rule in rules:
        key = (rule.rule_id, rule.source_id, rule.source_version, rule.source_sha256)
        approval = valid_rows.get(key)
        review_status = str(approval["review_status"]) if approval is not None else "not_approved"
        reviewed_rule = replace(rule, review_status=review_status)
        reviewed_rules.append(reviewed_rule)
        if approval is not None:
            matched_keys.add(key)

        source_key = (rule.source_id, rule.source_version, rule.source_sha256)
        source = source_rows.loc[source_key] if source_key in source_rows.index else None
        source_approved = source is not None and source["review_status"] == "approved"
        effective_date = pd.Timestamp(source["effective_date"]).date() if source is not None else None
        effective = effective_date is not None and effective_date <= as_of
        human_approved = review_status == "approved"
        governance_approved = (
            governance_eligible_source_ids is None
            or rule.source_id in governance_eligible_source_ids
        )
        compatibility_approved = (
            compatibility_eligible_source_ids is None
            or rule.source_id in compatibility_eligible_source_ids
        )
        candidate_reviewed = (
            candidate_reviewed_rule_keys is None
            or key in candidate_reviewed_rule_keys
        )
        source_approval_recorded = (
            source_approval_eligible_source_ids is None
            or rule.source_id in source_approval_eligible_source_ids
        )
        active = (
            human_approved
            and source_approved
            and effective
            and governance_approved
            and compatibility_approved
            and candidate_reviewed
            and source_approval_recorded
        )
        if active:
            active_rules.append(reviewed_rule)

        detail = (
            f"rule={rule.rule_id}; rule_status={review_status}; "
            f"source_status={source['review_status'] if source is not None else 'missing'}; "
            f"effective_date={effective_date}; governance_eligible={governance_approved}; "
            f"compatibility_eligible={compatibility_approved}; "
            f"candidate_reviewed={candidate_reviewed}"
            f"; source_approval_recorded={source_approval_recorded}"
        )
        checks.append(_check(
            "approved_source_required",
            rule.source_id,
            "PASS" if active else "WARN",
            "information" if active else "review_required",
            "Only a human-approved rule from an approved, effective, hash-matched source may become active",
            detail,
        ))
        checks.append(_check(
            "source_governance_required",
            rule.source_id,
            "PASS" if governance_approved else "WARN",
            "information" if governance_approved else "review_required",
            "Rule activation requires complete source ownership and authority governance",
            f"rule={rule.rule_id}; governance_eligible={governance_approved}",
        ))
        checks.append(_check(
            "source_compatibility_required",
            rule.source_id,
            "PASS" if compatibility_approved else "WARN",
            "information" if compatibility_approved else "review_required",
            "Rule activation requires a reviewed, technically compatible onboarding method",
            f"rule={rule.rule_id}; compatibility_eligible={compatibility_approved}",
        ))
        checks.append(_check(
            "candidate_review_required",
            rule.source_id,
            "PASS" if candidate_reviewed else "WARN",
            "information" if candidate_reviewed else "review_required",
            "Rule activation requires completed Phase 5 candidate review",
            f"rule={rule.rule_id}; candidate_reviewed={candidate_reviewed}",
        ))
        checks.append(_check(
            "source_approval_evidence_required",
            rule.source_id,
            "PASS" if source_approval_recorded else "WARN",
            "information" if source_approval_recorded else "review_required",
            "Rule activation requires hash-bound source approval evidence",
            f"rule={rule.rule_id}; source_approval_recorded={source_approval_recorded}",
        ))

    for key in sorted(set(valid_rows) - matched_keys):
        checks.append(_check(
            "approval_matches_candidate",
            key[1],
            "WARN",
            "review_required",
            "Approval record must match an extracted candidate exactly",
            f"rule={key[0]}; version={key[2]}; hash={key[3]}",
        ))
    check_frame = pd.DataFrame(checks, columns=QUALITY_COLUMNS)
    if check_frame["status"].eq("FAIL").any():
        active_rules = []
    return ApprovalResult(reviewed_rules, active_rules, check_frame)
