"""Consistency checks and review views for prepared EA guidance artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from cluster_c.config import (
    GUIDANCE_COMPATIBILITY_REPORT_PATH,
    GUIDANCE_CONTENT_PATH,
    GUIDANCE_QUALITY_PATH,
    GUIDANCE_RULES_PATH,
    GUIDANCE_SUMMARY_PATH,
)

from .decision_rules import GuidanceRegistry, load_guidance_registry


class GuidanceInspectionError(ValueError):
    """Prepared guidance artifacts are missing or mutually inconsistent."""


@dataclass(frozen=True)
class GuidanceInspectionResult:
    summary: dict[str, Any]
    sources: pd.DataFrame
    candidates: pd.DataFrame
    compatibility: pd.DataFrame
    failures: pd.DataFrame
    warnings: pd.DataFrame
    pages: pd.DataFrame
    registry: GuidanceRegistry


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GuidanceInspectionError(f"Unable to read {path}: {exc}") from exc


def inspect_guidance_artifacts(
    *,
    source_id: str | None = None,
    content_path: Path = GUIDANCE_CONTENT_PATH,
    quality_path: Path = GUIDANCE_QUALITY_PATH,
    rules_path: Path = GUIDANCE_RULES_PATH,
    compatibility_path: Path = GUIDANCE_COMPATIBILITY_REPORT_PATH,
    summary_path: Path = GUIDANCE_SUMMARY_PATH,
) -> GuidanceInspectionResult:
    """Load prepared outputs and verify that their counts agree."""
    required = [content_path, quality_path, rules_path, compatibility_path, summary_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise GuidanceInspectionError(f"Missing prepared guidance artifacts: {missing}")
    try:
        content = pd.read_parquet(content_path, engine="pyarrow")
        quality = pd.read_csv(quality_path, keep_default_na=False)
        compatibility = pd.read_csv(compatibility_path, keep_default_na=False)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise GuidanceInspectionError(f"Unable to load prepared guidance artifacts: {exc}") from exc
    summary = _read_json(summary_path)
    registry = load_guidance_registry(rules_path)

    actual = {
        "extracted_source_count": int(content["source_id"].nunique()) if not content.empty else 0,
        "extracted_page_count": len(content),
        "quality_check_count": len(quality),
        "quality_failure_count": int(quality["status"].eq("FAIL").sum()),
        "quality_warning_count": int(quality["status"].eq("WARN").sum()),
        "candidate_rule_count": len(registry.candidate_rules),
        "active_rule_count": len(registry.active_rules),
    }
    mismatches = {
        key: {"summary": summary.get(key), "actual": value}
        for key, value in actual.items()
        if summary.get(key) != value
    }
    if mismatches:
        raise GuidanceInspectionError(f"Prepared guidance artifact counts disagree: {mismatches}")

    source_rows = []
    for (item_source_id, version), group in content.groupby(["source_id", "source_version"], sort=True):
        source_rows.append({
            "source_id": item_source_id,
            "source_version": version,
            "review_status": group["review_status"].iloc[0],
            "page_count": len(group),
            "first_page": int(group["page_number"].min()),
            "last_page": int(group["page_number"].max()),
            "text_character_count": int(group["page_text"].str.len().sum()),
            "source_sha256": group["source_sha256"].iloc[0],
        })
    sources = pd.DataFrame(source_rows)
    candidates = pd.DataFrame([rule.to_dict() for rule in registry.candidate_rules])

    if source_id is not None:
        known_sources = set(content["source_id"].astype(str)) | set(compatibility["source_id"].astype(str))
        if source_id not in known_sources:
            raise GuidanceInspectionError(f"Unknown guidance source_id: {source_id}")
        content = content.loc[content.source_id == source_id]
        quality = quality.loc[quality.source_id.isin([source_id, "<RULE_SET>"])]
        compatibility = compatibility.loc[compatibility.source_id == source_id]
        if not sources.empty:
            sources = sources.loc[sources.source_id == source_id]
        if not candidates.empty:
            candidates = candidates.loc[candidates.source_id == source_id]

    return GuidanceInspectionResult(
        summary=summary,
        sources=sources.reset_index(drop=True),
        candidates=candidates.reset_index(drop=True),
        compatibility=compatibility.reset_index(drop=True),
        failures=quality.loc[quality.status == "FAIL"].reset_index(drop=True),
        warnings=quality.loc[quality.status == "WARN"].reset_index(drop=True),
        pages=content.sort_values(["source_id", "page_number"]).reset_index(drop=True),
        registry=registry,
    )
