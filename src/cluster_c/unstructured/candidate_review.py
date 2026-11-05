"""Hash-bound human review controls for extracted EA guidance candidates."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from cluster_c.config import (
    GUIDANCE_CANDIDATE_REVIEW_REPORT_PATH,
    GUIDANCE_CANDIDATE_REVIEWS_PATH,
    GUIDANCE_REVIEW_POLICY_PATH,
)

from .manifest import QUALITY_COLUMNS
from .schemas import GuidanceRule


CANDIDATE_REVIEW_COLUMNS = [
    "rule_id",
    "source_id",
    "source_version",
    "source_sha256",
    "candidate_sha256",
    "review_decision",
    "page_compared",
    "ambiguity_resolved",
    "measure_operator_confirmed",
    "scope_confirmed",
    "reviewed_by",
    "reviewed_at_utc",
    "review_notes",
]
REVIEW_POLICY_COLUMNS = [
    "policy_id",
    "status_precedence",
    "review_status",
    "reviewed_by",
    "reviewed_at_utc",
    "review_notes",
]
REVIEW_DECISIONS = {"in_review", "confirmed", "correction_required", "rejected"}
POLICY_REVIEW_STATUSES = {"in_review", "confirmed", "rejected"}
PRECEDENCE_POLICY_ID = "lens_status_precedence"
EXPECTED_STATUS_PRECEDENCE = "attention>unknown>favorable>neutral>report_only"
RULE_KEY_COLUMNS = ["rule_id", "source_id", "source_version", "source_sha256"]


@dataclass(frozen=True)
class CandidateReviewResult:
    report: pd.DataFrame
    checks: pd.DataFrame
    confirmed_rule_keys: set[tuple[str, str, str, str]]
    precedence_confirmed: bool

    @property
    def is_valid(self) -> bool:
        return not self.checks["status"].eq("FAIL").any()

    @property
    def is_complete(self) -> bool:
        if not self.is_valid or not self.precedence_confirmed:
            return False
        if self.report.empty:
            return True
        return self.report["review_complete"].all()


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


def candidate_sha256(rule: GuidanceRule) -> str:
    """Fingerprint every extracted field except the later approval status."""
    values = rule.to_dict()
    values.pop("review_status", None)
    canonical = json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _read_registry(path: Path, columns: list[str], label: str) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    checks: list[dict[str, str]] = []
    try:
        frame = pd.read_csv(path, dtype="string", keep_default_na=False)
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        checks.append(_check(
            f"{label}_readable",
            "<REVIEW>",
            "FAIL",
            "blocking",
            f"{label.replace('_', ' ').title()} must be a readable CSV",
            str(exc),
        ))
        return pd.DataFrame(columns=columns), checks
    missing = sorted(set(columns) - set(frame.columns))
    checks.append(_check(
        f"{label}_columns",
        "<REVIEW>",
        "PASS" if not missing else "FAIL",
        "blocking",
        f"{label.replace('_', ' ').title()} must contain all required columns",
        f"missing={missing}" if missing else "",
    ))
    return frame, checks


def _parse_bool(value: object) -> bool | None:
    normalized = str(value).strip().casefold()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    return None


def _reviewer_evidence(row: pd.Series) -> bool:
    reviewed_at = pd.to_datetime(row["reviewed_at_utc"], errors="coerce", utc=True)
    return bool(str(row["reviewed_by"]).strip()) and not pd.isna(reviewed_at) and bool(str(row["review_notes"]).strip())


def validate_candidate_reviews(
    rules: list[GuidanceRule],
    reviews_path: Path = GUIDANCE_CANDIDATE_REVIEWS_PATH,
    policy_path: Path = GUIDANCE_REVIEW_POLICY_PATH,
) -> CandidateReviewResult:
    """Validate reviews without changing extracted candidate values."""
    reviews, checks = _read_registry(reviews_path, CANDIDATE_REVIEW_COLUMNS, "candidate_review_registry")
    policy, policy_checks = _read_registry(policy_path, REVIEW_POLICY_COLUMNS, "review_policy_registry")
    checks.extend(policy_checks)
    report_rows: list[dict[str, object]] = []
    confirmed_keys: set[tuple[str, str, str, str]] = set()

    if set(CANDIDATE_REVIEW_COLUMNS).issubset(reviews.columns):
        duplicate_mask = reviews.duplicated(RULE_KEY_COLUMNS, keep=False)
        checks.append(_check(
            "candidate_review_key_unique",
            "<REVIEW>",
            "FAIL" if duplicate_mask.any() else "PASS",
            "blocking",
            "Each hash-bound candidate may have only one Phase 5 review record",
            f"duplicates={int(duplicate_mask.sum())}" if duplicate_mask.any() else "",
        ))
    else:
        duplicate_mask = pd.Series(False, index=reviews.index)

    review_by_key: dict[tuple[str, str, str, str], pd.Series] = {}
    if not duplicate_mask.any() and set(CANDIDATE_REVIEW_COLUMNS).issubset(reviews.columns):
        for _, row in reviews.iterrows():
            key = tuple(str(row[column]) for column in RULE_KEY_COLUMNS)
            review_by_key[key] = row

    policy_confirmed = False
    if set(REVIEW_POLICY_COLUMNS).issubset(policy.columns):
        precedence_rows = policy.loc[policy["policy_id"] == PRECEDENCE_POLICY_ID]
        unique_policy = len(precedence_rows) == 1
        checks.append(_check(
            "lens_status_precedence_record_unique",
            "<REVIEW_POLICY>",
            "PASS" if unique_policy else "FAIL",
            "blocking",
            "Exactly one lens-status precedence review record is required",
            f"count={len(precedence_rows)}",
        ))
        if unique_policy:
            row = precedence_rows.iloc[0]
            precedence_matches = row["status_precedence"] == EXPECTED_STATUS_PRECEDENCE
            status_allowed = row["review_status"] in POLICY_REVIEW_STATUSES
            evidence = _reviewer_evidence(row) if row["review_status"] in {"confirmed", "rejected"} else True
            structurally_valid = precedence_matches and status_allowed and evidence
            policy_confirmed = structurally_valid and row["review_status"] == "confirmed"
            checks.append(_check(
                "lens_status_precedence_review",
                "<REVIEW_POLICY>",
                "PASS" if policy_confirmed else ("FAIL" if not structurally_valid else "WARN"),
                "information" if policy_confirmed else ("blocking" if not structurally_valid else "review_required"),
                "EA owners must confirm the configured conservative lens-status precedence",
                f"precedence={row['status_precedence']}; status={row['review_status']}",
            ))

    candidate_keys = set()
    for rule in rules:
        key = (rule.rule_id, rule.source_id, rule.source_version, rule.source_sha256)
        candidate_keys.add(key)
        fingerprint = candidate_sha256(rule)
        review = review_by_key.get(key)
        decision = "missing"
        complete = False
        eligible = False
        detail = "No Phase 5 review record matches this candidate"
        review_values = {column: "" for column in CANDIDATE_REVIEW_COLUMNS[5:]}
        if review is not None:
            review_values = {column: str(review[column]) for column in CANDIDATE_REVIEW_COLUMNS[5:]}
            decision = str(review["review_decision"])
            booleans = {
                column: _parse_bool(review[column])
                for column in (
                    "page_compared",
                    "ambiguity_resolved",
                    "measure_operator_confirmed",
                    "scope_confirmed",
                )
            }
            fingerprint_matches = str(review["candidate_sha256"]) == fingerprint
            decision_allowed = decision in REVIEW_DECISIONS
            booleans_valid = all(value is not None for value in booleans.values())
            evidence = _reviewer_evidence(review) if decision != "in_review" else True
            record_valid = fingerprint_matches and decision_allowed and booleans_valid and evidence
            if decision == "confirmed":
                complete = record_valid and all(booleans.values())
                eligible = complete and policy_confirmed
            elif decision == "rejected":
                complete = record_valid and booleans["page_compared"] is True
            detail = (
                f"decision={decision}; fingerprint_matches={fingerprint_matches}; "
                f"review_controls={booleans}; reviewer_evidence={evidence}"
            )
            if not record_valid:
                checks.append(_check(
                    "candidate_review_record_valid",
                    rule.source_id,
                    "FAIL",
                    "blocking",
                    "Phase 5 review records must match the candidate fingerprint and contain valid controls",
                    f"rule={rule.rule_id}; {detail}",
                ))
        if eligible:
            confirmed_keys.add(key)
        status = "PASS" if complete else "WARN"
        checks.append(_check(
            "candidate_review_complete",
            rule.source_id,
            status,
            "information" if complete else "review_required",
            "Each candidate must be confirmed or explicitly rejected after comparison with its cited page",
            f"rule={rule.rule_id}; {detail}",
        ))
        report_rows.append({
            **rule.to_dict(),
            "candidate_sha256": fingerprint,
            **review_values,
            "review_complete": complete,
            "activation_review_eligible": eligible,
            "review_detail": detail,
        })

    for key in sorted(set(review_by_key) - candidate_keys):
        checks.append(_check(
            "candidate_review_matches_candidate",
            key[1],
            "WARN",
            "review_required",
            "Phase 5 review records must match a currently extracted candidate",
            f"rule={key[0]}; version={key[2]}; hash={key[3]}",
        ))

    return CandidateReviewResult(
        report=pd.DataFrame(report_rows),
        checks=pd.DataFrame(checks, columns=QUALITY_COLUMNS),
        confirmed_rule_keys=confirmed_keys,
        precedence_confirmed=policy_confirmed,
    )


def initialize_candidate_reviews(
    rules: list[GuidanceRule],
    reviews_path: Path = GUIDANCE_CANDIDATE_REVIEWS_PATH,
) -> int:
    """Append missing review rows while preserving all existing human decisions."""
    if reviews_path.exists():
        reviews = pd.read_csv(reviews_path, dtype="string", keep_default_na=False)
        missing = sorted(set(CANDIDATE_REVIEW_COLUMNS) - set(reviews.columns))
        if missing:
            raise ValueError(f"Candidate review registry is missing columns: {missing}")
    else:
        reviews = pd.DataFrame(columns=CANDIDATE_REVIEW_COLUMNS)
    existing = {
        tuple(str(row[column]) for column in RULE_KEY_COLUMNS)
        for _, row in reviews.iterrows()
    }
    rows = []
    for rule in rules:
        key = (rule.rule_id, rule.source_id, rule.source_version, rule.source_sha256)
        if key in existing:
            continue
        rows.append({
            "rule_id": rule.rule_id,
            "source_id": rule.source_id,
            "source_version": rule.source_version,
            "source_sha256": rule.source_sha256,
            "candidate_sha256": candidate_sha256(rule),
            "review_decision": "in_review",
            "page_compared": False,
            "ambiguity_resolved": False,
            "measure_operator_confirmed": False,
            "scope_confirmed": False,
            "reviewed_by": "",
            "reviewed_at_utc": "",
            "review_notes": "",
        })
    if not rows:
        return 0
    updated = pd.concat([reviews, pd.DataFrame(rows)], ignore_index=True)[CANDIDATE_REVIEW_COLUMNS]
    reviews_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = reviews_path.with_name(f".{reviews_path.name}.tmp")
    updated.to_csv(temporary, index=False)
    os.replace(temporary, reviews_path)
    return len(rows)


def save_candidate_review_report(
    result: CandidateReviewResult,
    path: Path = GUIDANCE_CANDIDATE_REVIEW_REPORT_PATH,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    result.report.to_csv(path, index=False)
    return path
