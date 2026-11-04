"""Deterministic candidate-rule extraction from controlled EA guidance text."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .manifest import QUALITY_COLUMNS
from .schemas import GuidanceRule, validate_rule_set


RULE_HEADING = re.compile(r"^([A-Z][A-Z0-9_-]{2,99}) - (.+)$")
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
LENS_BY_PREFIX = {
    "GROWTH-": "growth",
    "CUSTOMER-": "customer_value",
    "SIMPLICITY-": "simplicity_with_scale",
}


@dataclass(frozen=True)
class RuleExtractionResult:
    rules: list[GuidanceRule]
    checks: pd.DataFrame

    @property
    def is_valid(self) -> bool:
        return not self.checks["status"].eq("FAIL").any()


def _quality(
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


def _rule_blocks(page_text: str) -> list[tuple[str, str, list[str]]]:
    lines = [line.strip() for line in page_text.splitlines() if line.strip()]
    blocks: list[tuple[str, str, list[str]]] = []
    index = 0
    while index < len(lines):
        match = RULE_HEADING.fullmatch(lines[index])
        if not match or not any(match.group(1).startswith(prefix) for prefix in LENS_BY_PREFIX):
            index += 1
            continue
        rule_id, title = match.groups()
        block_lines: list[str] = []
        index += 1
        while index < len(lines):
            next_match = RULE_HEADING.fullmatch(lines[index])
            if next_match and any(next_match.group(1).startswith(prefix) for prefix in LENS_BY_PREFIX):
                break
            if lines[index] == "GOVERNANCE REQUIREMENTS":
                break
            block_lines.append(lines[index])
            index += 1
        blocks.append((rule_id, title, block_lines))
    return blocks


def _lens(rule_id: str) -> str:
    return next(lens for prefix, lens in LENS_BY_PREFIX.items() if rule_id.startswith(prefix))


def _comparison_from_favorable(line: str) -> tuple[str, float]:
    normalized = line.casefold()
    if "greater than or equal" in normalized:
        operator = "greater_than_or_equal"
    elif "less than or equal" in normalized:
        operator = "less_than_or_equal"
    elif "greater than" in normalized:
        operator = "greater_than"
    elif "less than" in normalized:
        operator = "less_than"
    elif "equal to" in normalized:
        operator = "equal"
    else:
        raise ValueError(f"Unsupported favorable comparison: {line}")
    values = NUMBER.findall(line)
    if not values:
        raise ValueError(f"Comparison threshold is missing: {line}")
    return operator, float(values[-1])


def _candidate_mapping(
    source: Any,
    page_number: int,
    rule_id: str,
    title: str,
    lines: list[str],
) -> tuple[dict[str, Any], bool]:
    measure_line = next((line for line in lines if line.startswith("Measure:")), None)
    if measure_line is None:
        raise ValueError("Measure line is missing")
    measure_expression = measure_line.split(":", 1)[1].strip()
    if " minus " in measure_expression:
        measure, comparison_measure = [value.strip() for value in measure_expression.split(" minus ", 1)]
    else:
        measure, comparison_measure = measure_expression, None

    report_line = next((line for line in lines if line.startswith("Report ")), None)
    favorable_line = next((line for line in lines if line.startswith("Favorable:")), None)
    attention_line = next((line for line in lines if line.startswith("Attention:")), None)
    neutral_line = next((line for line in lines if line.startswith("Neutral:")), None)
    explanation_line = next((line for line in lines if line.startswith("Explanation:")), None)
    scope_line = next((line for line in lines if line.startswith("Scope:")), None)

    if report_line:
        operator = "report_only"
        threshold = None
        status_if_matched = "report_only"
        status_if_not_matched = "report_only"
    elif favorable_line:
        operator, threshold = _comparison_from_favorable(favorable_line)
        status_if_matched = "favorable"
        status_if_not_matched = "attention" if attention_line else "unknown"
    else:
        raise ValueError("A report instruction or favorable comparison is required")

    explanation = (
        explanation_line.split(":", 1)[1].strip()
        if explanation_line
        else report_line or title
    )
    scope_type = "global"
    scope_column = None
    scope_value = None
    if scope_line:
        scope_expression = scope_line.split(":", 1)[1].strip()
        if scope_expression.casefold() != "global":
            if "=" not in scope_expression:
                raise ValueError("Scope must be 'global' or use COLUMN=VALUE")
            scope_column, scope_value = [value.strip() for value in scope_expression.split("=", 1)]
            scope_type = "portfolio_filter"

    mapping = {
        "rule_id": rule_id,
        "lens": _lens(rule_id),
        "measure": measure,
        "operator": operator,
        "threshold": threshold,
        "comparison_measure": comparison_measure,
        "status_if_matched": status_if_matched,
        "status_if_not_matched": status_if_not_matched,
        "explanation": explanation,
        "scope_type": scope_type,
        "scope_column": scope_column,
        "scope_value": scope_value,
        "source_id": source.source_id,
        "source_version": source.source_version,
        "source_page": page_number,
        "source_sha256": source.source_sha256,
        # Extraction creates a candidate only. A separate human approval
        # registry may move it to approved after review.
        "review_status": "not_approved",
    }
    return mapping, neutral_line is not None and attention_line is not None


def extract_candidate_rules(content: pd.DataFrame) -> RuleExtractionResult:
    """Extract schema-controlled candidates from the prepared page content."""
    rules: list[GuidanceRule] = []
    checks: list[dict[str, str]] = []
    for page in content.itertuples(index=False):
        text_present = bool(str(page.page_text).strip())
        checks.append(_quality(
            "page_text_present",
            page.source_id,
            "PASS" if text_present else "FAIL",
            "blocking",
            "Every prepared PDF page must contain extracted text",
            f"page={page.page_number}",
        ))
        if not text_present:
            continue
        for rule_id, title, lines in _rule_blocks(page.page_text):
            try:
                mapping, has_three_outcomes = _candidate_mapping(
                    page,
                    int(page.page_number),
                    rule_id,
                    title,
                    lines,
                )
                rule = GuidanceRule.from_mapping(mapping)
            except (TypeError, ValueError) as exc:
                checks.append(_quality(
                    "candidate_rule_schema_valid",
                    page.source_id,
                    "FAIL",
                    "blocking",
                    "Extracted candidates must match the controlled rule schema",
                    f"rule={rule_id}; error={exc}",
                ))
                continue
            rules.append(rule)
            checks.extend([
                _quality("candidate_rule_schema_valid", page.source_id, "PASS", "blocking", "Extracted candidates must match the controlled rule schema", f"rule={rule_id}"),
                _quality("lens_allowed", page.source_id, "PASS", "blocking", "Candidate lens must use the controlled vocabulary", f"rule={rule_id}; lens={rule.lens}"),
                _quality("measure_allowed", page.source_id, "PASS", "blocking", "Candidate measure must use the controlled vocabulary", f"rule={rule_id}; measure={rule.measure}"),
                _quality("operator_allowed", page.source_id, "PASS", "blocking", "Candidate operator must use the controlled vocabulary", f"rule={rule_id}; operator={rule.operator}"),
                _quality("threshold_valid", page.source_id, "PASS", "blocking", "Candidate threshold must be finite or not applicable", f"rule={rule_id}; threshold={rule.threshold}"),
                _quality("source_page_valid", page.source_id, "PASS", "blocking", "Candidate must retain a valid source page", f"rule={rule_id}; page={rule.source_page}"),
                _quality("review_status_valid", page.source_id, "PASS", "blocking", "Candidate review status must use the controlled vocabulary", f"rule={rule_id}; status={rule.review_status}"),
                _quality("scope_valid", page.source_id, "PASS", "blocking", "Candidate scope must be global or use an allowed portfolio filter", f"rule={rule_id}; scope={rule.scope_type}; column={rule.scope_column}; value={rule.scope_value}"),
            ])
            if has_three_outcomes:
                checks.append(_quality(
                    "multi_outcome_rule_requires_review",
                    page.source_id,
                    "WARN",
                    "review_required",
                    "Three-outcome source language requires human confirmation of the two-outcome candidate mapping",
                    f"rule={rule_id}",
                ))

    try:
        validate_rule_set(rules)
    except ValueError as exc:
        checks.append(_quality("rule_id_unique", "<RULE_SET>", "FAIL", "blocking", "Candidate rule IDs must be unique", str(exc)))
    else:
        checks.append(_quality("rule_id_unique", "<RULE_SET>", "PASS", "blocking", "Candidate rule IDs must be unique"))

    source_ids = sorted(set(content["source_id"])) if not content.empty else ["<RULE_SET>"]
    status = "PASS" if rules else "WARN"
    severity = "information" if rules else "review_required"
    checks.append(_quality(
        "candidate_rules_found",
        ",".join(source_ids),
        status,
        severity,
        "At least one controlled candidate rule should be extracted",
        f"count={len(rules)}",
    ))
    return RuleExtractionResult(rules, pd.DataFrame(checks, columns=QUALITY_COLUMNS))
