"""Controlled schema for candidate and approved EA guidance rules."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .manifest import ALLOWED_REVIEW_STATUSES, SHA256_PATTERN


ALLOWED_LENSES = {"growth", "customer_value", "simplicity_with_scale"}
ALLOWED_MEASURES = {
    "commitment_change",
    "commitment_change_pct",
    "npv_change",
    "average_term_years",
    "baseline_renewal_points",
    "scenario_renewal_points",
    "baseline_coverage_cv_pct",
    "scenario_coverage_cv_pct",
    "flexible_adoption_pct",
    "five_year_share_pct",
    "discount_rate_pct",
}
ALLOWED_OPERATORS = {
    "greater_than",
    "greater_than_or_equal",
    "less_than",
    "less_than_or_equal",
    "equal",
    "report_only",
}
ALLOWED_RULE_STATUSES = {
    "favorable",
    "neutral",
    "attention",
    "report_only",
    "not_applicable",
    "unknown",
}
ALLOWED_SCOPE_TYPES = {"global", "portfolio_filter"}
ALLOWED_SCOPE_COLUMNS = {"SCMS", "SALES_LEVEL_2", "SUITE_NAME"}
RULE_ID_PATTERN = re.compile(r"^[A-Z][A-Z0-9_-]{2,99}$")


@dataclass(frozen=True)
class GuidanceRule:
    rule_id: str
    lens: str
    measure: str
    operator: str
    threshold: float | None
    comparison_measure: str | None
    status_if_matched: str
    status_if_not_matched: str
    explanation: str
    scope_type: str
    scope_column: str | None
    scope_value: str | None
    source_id: str
    source_version: str
    source_page: int
    source_sha256: str
    review_status: str

    def __post_init__(self) -> None:
        errors = validate_rule_values(asdict(self))
        if errors:
            raise ValueError("Invalid guidance rule: " + "; ".join(errors))

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "GuidanceRule":
        expected = set(cls.__dataclass_fields__)
        missing = sorted(expected - set(values))
        extra = sorted(set(values) - expected)
        if missing or extra:
            raise ValueError(f"Guidance rule fields mismatch: missing={missing}; extra={extra}")
        normalized = dict(values)
        if normalized["threshold"] is not None:
            normalized["threshold"] = float(normalized["threshold"])
        normalized["source_page"] = int(normalized["source_page"])
        return cls(**normalized)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_rule_values(values: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    rule_id = str(values.get("rule_id", ""))
    lens = str(values.get("lens", ""))
    measure = str(values.get("measure", ""))
    operator = str(values.get("operator", ""))
    comparison_measure = values.get("comparison_measure")
    threshold = values.get("threshold")
    scope_type = str(values.get("scope_type", ""))
    scope_column = values.get("scope_column")
    scope_value = values.get("scope_value")

    if not RULE_ID_PATTERN.fullmatch(rule_id):
        errors.append("rule_id must use the controlled uppercase identifier format")
    if lens not in ALLOWED_LENSES:
        errors.append(f"lens must be one of {sorted(ALLOWED_LENSES)}")
    if measure not in ALLOWED_MEASURES:
        errors.append(f"measure must be one of {sorted(ALLOWED_MEASURES)}")
    if operator not in ALLOWED_OPERATORS:
        errors.append(f"operator must be one of {sorted(ALLOWED_OPERATORS)}")
    if comparison_measure is not None and comparison_measure not in ALLOWED_MEASURES:
        errors.append("comparison_measure must be null or an allowed measure")
    if threshold is not None:
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            errors.append("threshold must be null or numeric")
        elif not math.isfinite(float(threshold)):
            errors.append("threshold must be finite")
    if operator != "report_only" and threshold is None and comparison_measure is None:
        errors.append("comparison rules require a threshold or comparison_measure")
    if str(values.get("status_if_matched", "")) not in ALLOWED_RULE_STATUSES:
        errors.append("status_if_matched must use the controlled vocabulary")
    if str(values.get("status_if_not_matched", "")) not in ALLOWED_RULE_STATUSES:
        errors.append("status_if_not_matched must use the controlled vocabulary")
    if not str(values.get("explanation", "")).strip():
        errors.append("explanation must be populated")
    if scope_type not in ALLOWED_SCOPE_TYPES:
        errors.append(f"scope_type must be one of {sorted(ALLOWED_SCOPE_TYPES)}")
    elif scope_type == "global" and (scope_column is not None or scope_value is not None):
        errors.append("global rules must not define scope_column or scope_value")
    elif scope_type == "portfolio_filter":
        if scope_column not in ALLOWED_SCOPE_COLUMNS:
            errors.append(f"portfolio_filter scope_column must be one of {sorted(ALLOWED_SCOPE_COLUMNS)}")
        if not str(scope_value or "").strip():
            errors.append("portfolio_filter scope_value must be populated")
    if not str(values.get("source_id", "")).strip():
        errors.append("source_id must be populated")
    if not str(values.get("source_version", "")).strip():
        errors.append("source_version must be populated")
    source_page = values.get("source_page")
    if isinstance(source_page, bool) or not isinstance(source_page, int) or source_page < 1:
        errors.append("source_page must be a positive integer")
    if not SHA256_PATTERN.fullmatch(str(values.get("source_sha256", ""))):
        errors.append("source_sha256 must be a lowercase SHA-256 value")
    if str(values.get("review_status", "")) not in ALLOWED_REVIEW_STATUSES:
        errors.append("review_status must use the controlled vocabulary")
    return errors


def validate_rule_set(rules: list[GuidanceRule]) -> None:
    rule_ids = [rule.rule_id for rule in rules]
    duplicates = sorted({rule_id for rule_id in rule_ids if rule_ids.count(rule_id) > 1})
    if duplicates:
        raise ValueError(f"Guidance rule IDs must be unique; duplicates={duplicates}")
