"""Safe evaluation of approved EA guidance rules against scenario results."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from cluster_c.config import GUIDANCE_RULES_PATH

from .schemas import ALLOWED_LENSES, GuidanceRule, validate_rule_set


REGISTRY_SCHEMA_VERSION = "1.0"
STATUS_PRIORITY = {
    "attention": 5,
    "unknown": 4,
    "favorable": 3,
    "neutral": 2,
    "report_only": 1,
    "not_applicable": 0,
}


class GuidanceRegistryError(ValueError):
    """The generated guidance registry cannot be trusted or evaluated."""


@dataclass(frozen=True)
class GuidanceRegistry:
    schema_version: str
    generated_at_utc: str
    candidate_rules: list[GuidanceRule]
    active_rules: list[GuidanceRule]


def load_guidance_registry(path: Path = GUIDANCE_RULES_PATH) -> GuidanceRegistry:
    """Load and revalidate the generated, approval-gated rule registry."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GuidanceRegistryError(f"Unable to read EA guidance registry {path}: {exc}") from exc

    if payload.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise GuidanceRegistryError(
            f"Unsupported EA guidance registry schema: {payload.get('schema_version')!r}"
        )
    raw_rules = payload.get("rules")
    active_rule_ids = payload.get("active_rule_ids")
    if not isinstance(raw_rules, list) or not isinstance(active_rule_ids, list):
        raise GuidanceRegistryError("EA guidance registry must contain rules and active_rule_ids lists")

    try:
        candidates = [GuidanceRule.from_mapping(values) for values in raw_rules]
        validate_rule_set(candidates)
    except (TypeError, ValueError) as exc:
        raise GuidanceRegistryError(f"EA guidance registry contains invalid rules: {exc}") from exc

    if payload.get("candidate_rule_count") != len(candidates):
        raise GuidanceRegistryError("EA guidance candidate count does not match the registry contents")
    if len(active_rule_ids) != len(set(active_rule_ids)):
        raise GuidanceRegistryError("EA guidance active rule IDs must be unique")
    by_id = {rule.rule_id: rule for rule in candidates}
    unknown_active_ids = sorted(set(active_rule_ids) - set(by_id))
    if unknown_active_ids:
        raise GuidanceRegistryError(f"Unknown active EA guidance rule IDs: {unknown_active_ids}")
    active_rules = [by_id[rule_id] for rule_id in active_rule_ids]
    if any(rule.review_status != "approved" for rule in active_rules):
        raise GuidanceRegistryError("Every active EA guidance rule must have approved review status")
    if payload.get("active_rule_count") != len(active_rules):
        raise GuidanceRegistryError("EA guidance active count does not match active_rule_ids")
    return GuidanceRegistry(
        schema_version=payload["schema_version"],
        generated_at_utc=str(payload.get("generated_at_utc", "")),
        candidate_rules=candidates,
        active_rules=active_rules,
    )


def _scope_applies(rule: GuidanceRule, filter_context: Mapping[str, Sequence[Any]]) -> bool:
    if rule.scope_type == "global":
        return True
    selected = filter_context.get(str(rule.scope_column), [])
    if isinstance(selected, (str, int, float)):
        selected = [selected]
    # A segment-scoped rule applies only when the UI scope is exclusively that
    # segment. Applying it to a mixed or unfiltered portfolio would overreach.
    return {str(value) for value in selected} == {str(rule.scope_value)}


def _numeric_value(values: Mapping[str, Any], name: str) -> float:
    if name not in values:
        raise KeyError(name)
    value = float(values[name])
    if not math.isfinite(value):
        raise ValueError(f"{name} is not finite")
    return value


def _matches(operator: str, value: float, threshold: float) -> bool:
    comparisons = {
        "greater_than": lambda: value > threshold,
        "greater_than_or_equal": lambda: value >= threshold,
        "less_than": lambda: value < threshold,
        "less_than_or_equal": lambda: value <= threshold,
        "equal": lambda: math.isclose(value, threshold, rel_tol=1e-9, abs_tol=1e-9),
    }
    return comparisons[operator]()


def evaluate_guidance(
    summary: Mapping[str, Any],
    scenario_settings: Mapping[str, Any],
    rules: Sequence[GuidanceRule],
    filter_context: Mapping[str, Sequence[Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate approved rules using predefined comparisons only."""
    values = {**summary, **scenario_settings}
    filter_context = filter_context or {}
    evaluations: list[dict[str, Any]] = []
    for rule in rules:
        if rule.review_status != "approved":
            raise ValueError(f"Unapproved guidance rule cannot be evaluated: {rule.rule_id}")
        if not _scope_applies(rule, filter_context):
            continue
        comparison_value: float | None = None
        try:
            measure_value = _numeric_value(values, rule.measure)
            if rule.comparison_measure:
                comparison_value = _numeric_value(values, rule.comparison_measure)
            evaluated_value = measure_value - comparison_value if comparison_value is not None else measure_value
            if rule.operator == "report_only":
                status = "report_only"
                matched: bool | None = None
            else:
                if rule.threshold is None:
                    raise ValueError("comparison threshold is missing")
                matched = _matches(rule.operator, evaluated_value, rule.threshold)
                status = rule.status_if_matched if matched else rule.status_if_not_matched
            reason = rule.explanation
        except (KeyError, TypeError, ValueError) as exc:
            measure_value = None
            evaluated_value = None
            matched = None
            status = "unknown"
            reason = f"Rule could not be evaluated: {exc}"

        evaluations.append({
            "rule_id": rule.rule_id,
            "lens": rule.lens,
            "status": status,
            "reason": reason,
            "measure": rule.measure,
            "measure_value": measure_value,
            "comparison_measure": rule.comparison_measure,
            "comparison_value": comparison_value,
            "evaluated_value": evaluated_value,
            "operator": rule.operator,
            "threshold": rule.threshold,
            "matched": matched,
            "source_id": rule.source_id,
            "source_version": rule.source_version,
            "source_page": rule.source_page,
            "source_sha256": rule.source_sha256,
        })

    lenses: dict[str, dict[str, Any]] = {}
    for lens in sorted(ALLOWED_LENSES):
        lens_results = [result for result in evaluations if result["lens"] == lens]
        if not lens_results:
            lenses[lens] = {
                "status": "not_available",
                "reasons": [],
                "rule_ids": [],
                "guidance_versions": [],
            }
            continue
        overall = max(lens_results, key=lambda result: STATUS_PRIORITY[result["status"]])["status"]
        lenses[lens] = {
            "status": overall,
            "reasons": [result["reason"] for result in lens_results],
            "rule_ids": [result["rule_id"] for result in lens_results],
            "guidance_versions": sorted({result["source_version"] for result in lens_results}),
        }
    return {
        "applied": bool(evaluations),
        "active_rule_count": len(rules),
        "applied_rule_count": len(evaluations),
        "lenses": lenses,
        "rule_results": evaluations,
    }
