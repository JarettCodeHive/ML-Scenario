"""Preparation pipeline for versioned general EA guidance PDFs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pypdf

from cluster_c.config import (
    GUIDANCE_CANDIDATE_REVIEW_REPORT_PATH,
    GUIDANCE_COMPATIBILITY_PATH,
    GUIDANCE_COMPATIBILITY_REPORT_PATH,
    GUIDANCE_CONTENT_PATH,
    GUIDANCE_MANIFEST_PATH,
    GUIDANCE_PDF_DIR,
    GUIDANCE_QUALITY_PATH,
    GUIDANCE_RULES_PATH,
    GUIDANCE_SUMMARY_PATH,
    PROJECT_ROOT,
)

from .manifest import QUALITY_COLUMNS, ManifestValidationResult, validate_guidance_manifest
from .compatibility import assess_guidance_compatibility
from .pdf_extractor import PdfExtractionError, extract_pdf_pages
from .rule_extractor import extract_candidate_rules
from .schemas import GuidanceRule


CONTENT_COLUMNS = [
    "source_id",
    "source_path",
    "source_title",
    "source_version",
    "source_sha256",
    "effective_date",
    "review_status",
    "is_synthetic",
    "page_number",
    "page_text",
    "extractor_name",
    "extractor_version",
    "extracted_at_utc",
]


@dataclass(frozen=True)
class GuidancePreparationResult:
    content: pd.DataFrame
    compatibility: pd.DataFrame
    candidate_review: pd.DataFrame
    quality: pd.DataFrame
    candidate_rules: list[GuidanceRule]
    active_rules: list[GuidanceRule]
    generated_at_utc: datetime
    summary: dict[str, Any]

    @property
    def is_valid(self) -> bool:
        return not self.quality["status"].eq("FAIL").any()


def normalize_page_text(text: str) -> str:
    """Normalize spacing while retaining meaningful line boundaries."""
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    normalized: list[str] = []
    for line in lines:
        if line or (normalized and normalized[-1]):
            normalized.append(line)
    return "\n".join(normalized).strip()


def prepare_guidance(
    manifest_path: Path = GUIDANCE_MANIFEST_PATH,
    project_root: Path = PROJECT_ROOT,
    guidance_pdf_dir: Path = GUIDANCE_PDF_DIR,
    compatibility_path: Path = GUIDANCE_COMPATIBILITY_PATH,
    extracted_at: datetime | None = None,
) -> GuidancePreparationResult:
    """Validate PDFs and directly activate every structurally valid rule."""
    manifest_result: ManifestValidationResult = validate_guidance_manifest(
        manifest_path=manifest_path,
        project_root=project_root,
        guidance_pdf_dir=guidance_pdf_dir,
    )
    quality_rows = manifest_result.checks.to_dict("records")
    content_rows: list[dict[str, Any]] = []
    extracted_at = extracted_at or datetime.now(timezone.utc)
    compatibility_report = pd.DataFrame()

    if manifest_result.is_valid:
        compatibility = assess_guidance_compatibility(
            manifest_result.manifest,
            compatibility_path=compatibility_path,
            as_of=extracted_at.date(),
        )
        quality_rows.extend(compatibility.checks.to_dict("records"))
        compatibility_report = compatibility.report

    if manifest_result.is_valid and not any(row["status"] == "FAIL" for row in quality_rows):
        for row in manifest_result.manifest.itertuples(index=False):
            source_path = Path(row.resolved_path)
            try:
                pages = extract_pdf_pages(source_path)
            except PdfExtractionError as exc:
                quality_rows.append({
                    "check_id": "pdf_text_extractable",
                    "source_id": row.source_id,
                    "status": "FAIL",
                    "severity": "blocking",
                    "description": "PDF must contain extractable page-level text",
                    "detail": f"{exc.status}: {exc.detail}",
                })
                continue

            quality_rows.append({
                "check_id": "pdf_text_extractable",
                "source_id": row.source_id,
                "status": "PASS",
                "severity": "blocking",
                "description": "PDF must contain extractable page-level text",
                "detail": f"pages={len(pages)}",
            })
            for page in pages:
                content_rows.append({
                    "source_id": row.source_id,
                    "source_path": row.source_path,
                    "source_title": row.title,
                    "source_version": row.version,
                    "source_sha256": row.sha256,
                    "effective_date": row.effective_date,
                    "review_status": row.review_status,
                    "is_synthetic": row.is_synthetic,
                    "page_number": page.page_number,
                    "page_text": normalize_page_text(page.text),
                    "extractor_name": "pypdf",
                    "extractor_version": pypdf.__version__,
                    "extracted_at_utc": extracted_at,
                })

    content = pd.DataFrame(content_rows, columns=CONTENT_COLUMNS)
    base_quality = pd.DataFrame(quality_rows, columns=QUALITY_COLUMNS)
    candidate_rules: list[GuidanceRule] = []
    active_rules: list[GuidanceRule] = []
    candidate_review_report = pd.DataFrame(columns=["rule_id"])
    if not base_quality["status"].eq("FAIL").any() and not content.empty:
        extraction = extract_candidate_rules(content)
        quality_rows.extend(extraction.checks.to_dict("records"))
        if extraction.is_valid:
            # POC behavior: a valid registered PDF is trusted as the controlled
            # rule source. Human review/approval registries are intentionally not
            # consulted. ``approved`` is retained only as the legacy schema value
            # that identifies an executable rule to existing UI/evaluator code.
            candidate_rules = [replace(rule, review_status="approved") for rule in extraction.rules]
            active_rules = list(candidate_rules)

    quality = pd.DataFrame(quality_rows, columns=QUALITY_COLUMNS)
    failed_checks = int(quality["status"].eq("FAIL").sum())
    warning_checks = int(quality["status"].eq("WARN").sum())
    summary = {
        "source_count": len(manifest_result.manifest),
        "extracted_source_count": int(content["source_id"].nunique()) if not content.empty else 0,
        "extracted_page_count": len(content),
        "quality_check_count": len(quality),
        "quality_failure_count": failed_checks,
        "quality_warning_count": warning_checks,
        "candidate_rule_count": len(candidate_rules),
        "active_rule_count": len(active_rules),
        "activation_mode": "direct_poc",
        "candidate_review_complete_count": 0,
        "candidate_review_eligible_count": 0,
        "lens_status_precedence_confirmed": False,
        "valid": failed_checks == 0,
    }
    return GuidancePreparationResult(
        content=content,
        compatibility=compatibility_report,
        candidate_review=candidate_review_report,
        quality=quality,
        candidate_rules=candidate_rules,
        active_rules=active_rules,
        generated_at_utc=extracted_at,
        summary=summary,
    )


def save_guidance_preparation(
    result: GuidancePreparationResult,
    content_path: Path = GUIDANCE_CONTENT_PATH,
    quality_path: Path = GUIDANCE_QUALITY_PATH,
    rules_path: Path = GUIDANCE_RULES_PATH,
    compatibility_report_path: Path = GUIDANCE_COMPATIBILITY_REPORT_PATH,
    summary_path: Path = GUIDANCE_SUMMARY_PATH,
    candidate_review_report_path: Path = GUIDANCE_CANDIDATE_REVIEW_REPORT_PATH,
) -> dict[str, Path]:
    """Persist page content and the full preparation quality report."""
    content_path.parent.mkdir(parents=True, exist_ok=True)
    quality_path.parent.mkdir(parents=True, exist_ok=True)
    rules_path.parent.mkdir(parents=True, exist_ok=True)
    compatibility_report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_review_report_path.parent.mkdir(parents=True, exist_ok=True)
    result.content.to_parquet(content_path, index=False, engine="pyarrow", compression="snappy")
    result.quality.to_csv(quality_path, index=False)
    result.compatibility.to_csv(compatibility_report_path, index=False)
    result.candidate_review.to_csv(candidate_review_report_path, index=False)
    registry = {
        "schema_version": "1.0",
        "generated_at_utc": result.generated_at_utc.isoformat(),
        "candidate_rule_count": len(result.candidate_rules),
        "active_rule_count": len(result.active_rules),
        "active_rule_ids": [rule.rule_id for rule in result.active_rules],
        "rules": [rule.to_dict() for rule in result.candidate_rules],
    }
    rules_path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    summary_payload = {
        "generated_at_utc": result.generated_at_utc.isoformat(),
        **result.summary,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")
    return {
        "content": content_path,
        "quality": quality_path,
        "rules": rules_path,
        "compatibility": compatibility_report_path,
        "summary": summary_path,
        "candidate_review": candidate_review_report_path,
    }
