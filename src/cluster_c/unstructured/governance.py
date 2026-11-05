"""Ownership, authority, permitted-use, and lifecycle controls for guidance."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from cluster_c.config import GUIDANCE_GOVERNANCE_PATH

from .manifest import QUALITY_COLUMNS


GOVERNANCE_COLUMNS = [
    "source_id",
    "business_owner",
    "technical_owner",
    "authorized_rule_reviewer",
    "backup_reviewer",
    "source_repository",
    "governance_contact",
    "authority_status",
    "authority_confirmed_by",
    "authority_confirmed_at_utc",
    "storage_permitted",
    "text_extraction_permitted",
    "rule_derivation_permitted",
    "streamlit_reference_permitted",
    "retention_policy",
    "access_requirements",
    "retirement_date",
    "supersedes_source_id",
    "review_frequency_days",
]
OWNER_COLUMNS = [
    "business_owner",
    "technical_owner",
    "authorized_rule_reviewer",
    "backup_reviewer",
    "source_repository",
    "governance_contact",
]
PERMISSION_COLUMNS = [
    "storage_permitted",
    "text_extraction_permitted",
    "rule_derivation_permitted",
    "streamlit_reference_permitted",
]
ALLOWED_AUTHORITY_STATUSES = {"draft", "in_review", "approved", "retired", "superseded"}


@dataclass(frozen=True)
class GovernanceValidationResult:
    governance: pd.DataFrame
    checks: pd.DataFrame
    eligible_source_ids: set[str]

    @property
    def is_valid(self) -> bool:
        return not self.checks["status"].eq("FAIL").any()


def _parse_boolean(value: Any) -> bool | None:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def validate_guidance_governance(
    manifest: pd.DataFrame,
    governance_path: Path = GUIDANCE_GOVERNANCE_PATH,
    as_of: date | None = None,
) -> GovernanceValidationResult:
    """Validate governance records and identify sources eligible for activation."""
    as_of = as_of or datetime.now(timezone.utc).date()
    checks: list[dict[str, str]] = []

    def add(
        check_id: str,
        source_id: str,
        status: str,
        severity: str,
        description: str,
        detail: str = "",
    ) -> None:
        checks.append({
            "check_id": check_id,
            "source_id": source_id,
            "status": status,
            "severity": severity,
            "description": description,
            "detail": detail,
        })

    try:
        governance = pd.read_csv(governance_path, dtype="string", keep_default_na=False)
    except (OSError, pd.errors.ParserError, UnicodeError) as exc:
        add("governance_registry_readable", "<GOVERNANCE>", "FAIL", "blocking", "Governance registry must be a readable CSV", str(exc))
        return GovernanceValidationResult(
            pd.DataFrame(columns=GOVERNANCE_COLUMNS),
            pd.DataFrame(checks, columns=QUALITY_COLUMNS),
            set(),
        )

    missing = sorted(set(GOVERNANCE_COLUMNS) - set(governance.columns))
    add(
        "governance_registry_columns",
        "<GOVERNANCE>",
        "PASS" if not missing else "FAIL",
        "blocking",
        "Governance registry must contain all required columns",
        f"missing={missing}" if missing else "",
    )
    if missing:
        return GovernanceValidationResult(governance, pd.DataFrame(checks, columns=QUALITY_COLUMNS), set())

    source_ids = governance["source_id"].str.strip()
    duplicate_ids = sorted(source_ids[source_ids.duplicated(keep=False) & source_ids.ne("")].unique())
    add(
        "governance_source_id_unique",
        "<GOVERNANCE>",
        "PASS" if source_ids.ne("").all() and not duplicate_ids else "FAIL",
        "blocking",
        "Governance source IDs must be populated and unique",
        f"duplicates={duplicate_ids}" if duplicate_ids else "",
    )
    governance = governance.copy()
    governance["source_id"] = source_ids
    by_source = {row.source_id: row for row in governance.itertuples(index=False)}
    eligible: set[str] = set()

    for source in manifest.itertuples(index=False):
        source_id = str(source.source_id)
        record = by_source.get(source_id)
        source_approved = source.review_status == "approved"
        if record is None:
            add(
                "governance_record_present",
                source_id,
                "FAIL" if source_approved else "WARN",
                "blocking" if source_approved else "review_required",
                "Every guidance source requires an ownership and authority record",
            )
            continue
        add("governance_record_present", source_id, "PASS", "blocking", "Every guidance source requires an ownership and authority record")

        missing_owners = [column for column in OWNER_COLUMNS if not str(getattr(record, column)).strip()]
        owners_complete = not missing_owners
        add(
            "governance_owners_complete",
            source_id,
            "PASS" if owners_complete else ("FAIL" if source_approved else "WARN"),
            "blocking" if source_approved else "review_required",
            "Business, technical, reviewer, repository, and contact ownership must be explicit",
            f"missing={missing_owners}" if missing_owners else "",
        )

        authority_valid = record.authority_status in ALLOWED_AUTHORITY_STATUSES
        confirmation_time = pd.to_datetime(record.authority_confirmed_at_utc, errors="coerce") if record.authority_confirmed_at_utc else pd.NaT
        confirmation_complete = bool(record.authority_confirmed_by.strip()) and not pd.isna(confirmation_time)
        authority_approved = record.authority_status == "approved" and confirmation_complete
        add(
            "governance_authority_confirmed",
            source_id,
            "PASS" if authority_valid and authority_approved else ("FAIL" if source_approved else "WARN"),
            "blocking" if source_approved else "review_required",
            "Approved authority requires a confirmer and confirmation timestamp",
            f"authority_status={record.authority_status}; confirmed_by={record.authority_confirmed_by!r}",
        )

        parsed_permissions = {column: _parse_boolean(getattr(record, column)) for column in PERMISSION_COLUMNS}
        permissions_well_formed = all(value is not None for value in parsed_permissions.values())
        permissions_granted = permissions_well_formed and all(parsed_permissions.values())
        add(
            "governance_permitted_uses",
            source_id,
            "PASS" if permissions_granted else ("FAIL" if source_approved or not permissions_well_formed else "WARN"),
            "blocking" if source_approved or not permissions_well_formed else "review_required",
            "Storage, extraction, rule derivation, and Streamlit reference must be explicitly permitted",
            "; ".join(f"{key}={value}" for key, value in parsed_permissions.items()),
        )

        retention_complete = bool(record.retention_policy.strip()) and bool(record.access_requirements.strip())
        add(
            "governance_retention_access_confirmed",
            source_id,
            "PASS" if retention_complete else ("FAIL" if source_approved else "WARN"),
            "blocking" if source_approved else "review_required",
            "Retention policy and access requirements must be recorded",
        )

        retirement_date = pd.to_datetime(record.retirement_date, errors="coerce") if record.retirement_date else pd.NaT
        retirement_valid = not record.retirement_date or not pd.isna(retirement_date)
        active_lifecycle = pd.isna(retirement_date) or retirement_date.date() > as_of
        try:
            review_frequency_days = int(record.review_frequency_days)
        except (TypeError, ValueError):
            review_frequency_days = 0
        lifecycle_valid = retirement_valid and review_frequency_days > 0
        lifecycle_acceptable = lifecycle_valid and (not source_approved or active_lifecycle)
        add(
            "governance_lifecycle_valid",
            source_id,
            "PASS" if lifecycle_acceptable else "FAIL",
            "blocking",
            "Retirement date must be valid, approved sources must be active, and review frequency must be positive",
            f"retirement_date={record.retirement_date!r}; review_frequency_days={record.review_frequency_days!r}",
        )

        review_current = False
        if confirmation_complete and review_frequency_days > 0:
            next_review = confirmation_time.date() + timedelta(days=review_frequency_days)
            review_current = next_review >= as_of
        add(
            "governance_review_current",
            source_id,
            "PASS" if review_current else ("FAIL" if source_approved else "WARN"),
            "blocking" if source_approved else "review_required",
            "Authority confirmation must remain within the configured review interval",
        )

        source_eligible = (
            source_approved
            and owners_complete
            and authority_approved
            and permissions_granted
            and retention_complete
            and lifecycle_acceptable
            and active_lifecycle
            and review_current
        )
        add(
            "governance_activation_eligible",
            source_id,
            "PASS" if source_eligible else "WARN",
            "information" if source_eligible else "review_required",
            "Only a fully governed source is eligible to activate rules",
            f"eligible={source_eligible}",
        )
        if source_eligible:
            eligible.add(source_id)

    manifest_ids = set(manifest["source_id"].astype(str))
    for orphan_id in sorted(set(by_source) - manifest_ids):
        add(
            "governance_matches_source",
            orphan_id,
            "WARN",
            "review_required",
            "Governance record does not match a current source manifest entry",
        )
    return GovernanceValidationResult(
        governance,
        pd.DataFrame(checks, columns=QUALITY_COLUMNS),
        eligible,
    )
