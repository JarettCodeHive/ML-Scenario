"""Technical and reviewed onboarding compatibility for EA guidance PDFs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from cluster_c.config import GUIDANCE_COMPATIBILITY_PATH

from .manifest import QUALITY_COLUMNS
from .pdf_extractor import PdfExtractionError, extract_pdf_pages


COMPATIBILITY_COLUMNS = [
    "source_id",
    "selected_onboarding_method",
    "assessed_by",
    "assessed_at_utc",
    "assessment_notes",
]
REPORT_COLUMNS = [
    "source_id",
    "source_path",
    "pdf_status",
    "page_count",
    "text_character_count",
    "controlled_rule_count",
    "format_compatibility",
    "recommended_onboarding_method",
    "selected_onboarding_method",
    "assessed_by",
    "assessed_at_utc",
    "activation_compatible",
    "detail",
]
ALLOWED_ONBOARDING_METHODS = {
    "direct_controlled_pdf",
    "controlled_companion_pdf",
    "manual_rule_registry",
    "ocr_required",
    "rejected",
}
RULE_HEADING = re.compile(r"^[A-Z][A-Z0-9_-]{2,99} - .+$")


@dataclass(frozen=True)
class CompatibilityResult:
    report: pd.DataFrame
    checks: pd.DataFrame
    eligible_source_ids: set[str]

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


def _controlled_rule_count(page_texts: list[str]) -> int:
    count = 0
    for text in page_texts:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for index, line in enumerate(lines):
            if not RULE_HEADING.fullmatch(line):
                continue
            following = lines[index + 1:index + 8]
            has_measure = any(value.startswith("Measure:") for value in following)
            has_decision = any(
                value.startswith(("Favorable:", "Report "))
                for value in following
            )
            if has_measure and has_decision:
                count += 1
    return count


def assess_guidance_compatibility(
    manifest: pd.DataFrame,
    compatibility_path: Path = GUIDANCE_COMPATIBILITY_PATH,
    as_of: date | None = None,
) -> CompatibilityResult:
    """Classify source PDFs and validate the reviewed onboarding selection."""
    as_of = as_of or datetime.now(timezone.utc).date()
    checks: list[dict[str, str]] = []
    report_rows: list[dict[str, object]] = []
    try:
        decisions = pd.read_csv(compatibility_path, dtype="string", keep_default_na=False)
    except (OSError, pd.errors.ParserError, UnicodeError) as exc:
        checks.append(_check("compatibility_registry_readable", "<COMPATIBILITY>", "FAIL", "blocking", "Compatibility registry must be a readable CSV", str(exc)))
        return CompatibilityResult(
            pd.DataFrame(columns=REPORT_COLUMNS),
            pd.DataFrame(checks, columns=QUALITY_COLUMNS),
            set(),
        )

    missing = sorted(set(COMPATIBILITY_COLUMNS) - set(decisions.columns))
    checks.append(_check(
        "compatibility_registry_columns",
        "<COMPATIBILITY>",
        "PASS" if not missing else "FAIL",
        "blocking",
        "Compatibility registry must contain all required columns",
        f"missing={missing}" if missing else "",
    ))
    if missing:
        return CompatibilityResult(decisions, pd.DataFrame(checks, columns=QUALITY_COLUMNS), set())

    source_ids = decisions["source_id"].str.strip()
    duplicate_ids = sorted(source_ids[source_ids.duplicated(keep=False) & source_ids.ne("")].unique())
    checks.append(_check(
        "compatibility_source_id_unique",
        "<COMPATIBILITY>",
        "PASS" if source_ids.ne("").all() and not duplicate_ids else "FAIL",
        "blocking",
        "Compatibility source IDs must be populated and unique",
        f"duplicates={duplicate_ids}" if duplicate_ids else "",
    ))
    decisions = decisions.copy()
    decisions["source_id"] = source_ids
    by_source = {row.source_id: row for row in decisions.itertuples(index=False)}
    eligible: set[str] = set()

    for source in manifest.itertuples(index=False):
        source_id = str(source.source_id)
        source_approved = source.review_status == "approved"
        page_texts: list[str] = []
        pdf_status = "text_pdf"
        detail = ""
        try:
            pages = extract_pdf_pages(Path(source.resolved_path))
            page_texts = [page.text for page in pages]
            page_count = len(pages)
        except PdfExtractionError as exc:
            page_count = 0
            pdf_status = exc.status
            detail = exc.detail

        character_count = sum(len(text) for text in page_texts)
        rule_count = _controlled_rule_count(page_texts)
        if pdf_status == "text_pdf" and rule_count:
            format_compatibility = "controlled_rules"
            recommendation = "direct_controlled_pdf"
        elif pdf_status == "text_pdf":
            format_compatibility = "free_form_text"
            recommendation = "controlled_companion_pdf"
        elif pdf_status == "ocr_required":
            format_compatibility = "image_only"
            recommendation = "ocr_required"
        else:
            format_compatibility = "unsupported"
            recommendation = "rejected"

        decision = by_source.get(source_id)
        decision_present = decision is not None
        checks.append(_check(
            "compatibility_decision_present",
            source_id,
            "PASS" if decision_present else ("FAIL" if source_approved else "WARN"),
            "blocking" if source_approved else "review_required",
            "Every source requires a reviewed onboarding-method selection",
        ))
        selected_method = decision.selected_onboarding_method if decision else ""
        method_allowed = selected_method in ALLOWED_ONBOARDING_METHODS
        assessor = decision.assessed_by if decision else ""
        assessed_at = pd.to_datetime(decision.assessed_at_utc, errors="coerce") if decision and decision.assessed_at_utc else pd.NaT
        review_complete = bool(
            decision
            and decision.assessed_by.strip()
            and not pd.isna(assessed_at)
            and assessed_at.date() <= as_of
        )
        checks.append(_check(
            "compatibility_selection_valid",
            source_id,
            "PASS" if decision_present and method_allowed and review_complete else ("FAIL" if source_approved else "WARN"),
            "blocking" if source_approved else "review_required",
            "Onboarding selection requires an allowed method, assessor, and non-future timestamp",
            f"selected={selected_method!r}; assessed_by={assessor!r}",
        ))

        selection_matches = selected_method == recommendation
        checks.append(_check(
            "compatibility_selection_matches_assessment",
            source_id,
            "PASS" if selection_matches else "WARN",
            "information" if selection_matches else "review_required",
            "Selected onboarding method should match the technical assessment",
            f"recommended={recommendation}; selected={selected_method or '<MISSING>'}",
        ))
        activation_compatible = (
            pdf_status == "text_pdf"
            and format_compatibility == "controlled_rules"
            and selected_method == "direct_controlled_pdf"
            and method_allowed
            and review_complete
        )
        checks.append(_check(
            "document_activation_compatible",
            source_id,
            "PASS" if activation_compatible else "WARN",
            "information" if activation_compatible else "review_required",
            "Only reviewed controlled text PDFs are currently activation-compatible",
            f"pdf_status={pdf_status}; format={format_compatibility}; selected={selected_method or '<MISSING>'}",
        ))
        if activation_compatible:
            eligible.add(source_id)

        report_rows.append({
            "source_id": source_id,
            "source_path": source.source_path,
            "pdf_status": pdf_status,
            "page_count": page_count,
            "text_character_count": character_count,
            "controlled_rule_count": rule_count,
            "format_compatibility": format_compatibility,
            "recommended_onboarding_method": recommendation,
            "selected_onboarding_method": selected_method,
            "assessed_by": decision.assessed_by if decision else "",
            "assessed_at_utc": assessed_at,
            "activation_compatible": activation_compatible,
            "detail": detail,
        })

    manifest_ids = set(manifest["source_id"].astype(str))
    for orphan_id in sorted(set(by_source) - manifest_ids):
        checks.append(_check(
            "compatibility_matches_source",
            orphan_id,
            "WARN",
            "review_required",
            "Compatibility record does not match a current source manifest entry",
        ))
    return CompatibilityResult(
        pd.DataFrame(report_rows, columns=REPORT_COLUMNS),
        pd.DataFrame(checks, columns=QUALITY_COLUMNS),
        eligible,
    )
