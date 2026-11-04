from datetime import date
from pathlib import Path

import pandas as pd
from pypdf import PdfWriter

from cluster_c.config import GUIDANCE_COMPATIBILITY_PATH
from cluster_c.unstructured.compatibility import assess_guidance_compatibility
from cluster_c.unstructured.manifest import validate_guidance_manifest
from scripts.generate_sample_ea_guidance import build_pdf


def repository_manifest() -> pd.DataFrame:
    return validate_guidance_manifest().manifest.copy()


def write_decision(tmp_path: Path, method: str) -> Path:
    decisions = pd.read_csv(GUIDANCE_COMPATIBILITY_PATH, dtype="string", keep_default_na=False)
    decisions.loc[0, "selected_onboarding_method"] = method
    path = tmp_path / "guidance_compatibility.csv"
    decisions.to_csv(path, index=False)
    return path


def test_sample_pdf_is_directly_compatible() -> None:
    result = assess_guidance_compatibility(repository_manifest(), as_of=date(2026, 11, 4))

    assert result.is_valid
    assert result.eligible_source_ids == {"EA-GUIDE-SAMPLE-001"}
    row = result.report.iloc[0]
    assert row.pdf_status == "text_pdf"
    assert row.format_compatibility == "controlled_rules"
    assert row.recommended_onboarding_method == "direct_controlled_pdf"
    assert row.selected_onboarding_method == "direct_controlled_pdf"
    assert row.controlled_rule_count == 6
    assert bool(row.activation_compatible)


def test_free_form_text_recommends_controlled_companion(tmp_path: Path) -> None:
    pdf_path = tmp_path / "free_form.pdf"
    pdf_path.write_bytes(build_pdf([["EA Guidance Narrative", "This document has no controlled rule blocks."]]))
    manifest = repository_manifest()
    manifest.loc[0, "resolved_path"] = str(pdf_path)
    manifest.loc[0, "source_path"] = "free_form.pdf"
    decision_path = write_decision(tmp_path, "controlled_companion_pdf")

    result = assess_guidance_compatibility(manifest, decision_path, date(2026, 11, 4))

    assert result.is_valid
    assert result.eligible_source_ids == set()
    row = result.report.iloc[0]
    assert row.format_compatibility == "free_form_text"
    assert row.recommended_onboarding_method == "controlled_companion_pdf"
    assert not bool(row.activation_compatible)


def test_image_only_pdf_reports_ocr_required(tmp_path: Path) -> None:
    pdf_path = tmp_path / "image_only.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with pdf_path.open("wb") as output:
        writer.write(output)
    manifest = repository_manifest()
    manifest.loc[0, "resolved_path"] = str(pdf_path)
    manifest.loc[0, "source_path"] = "image_only.pdf"
    decision_path = write_decision(tmp_path, "ocr_required")

    result = assess_guidance_compatibility(manifest, decision_path, date(2026, 11, 4))

    assert result.is_valid
    row = result.report.iloc[0]
    assert row.pdf_status == "ocr_required"
    assert row.format_compatibility == "image_only"
    assert row.recommended_onboarding_method == "ocr_required"
    assert not bool(row.activation_compatible)


def test_approved_source_with_invalid_method_is_blocking(tmp_path: Path) -> None:
    manifest = repository_manifest()
    manifest.loc[0, "review_status"] = "approved"
    decision_path = write_decision(tmp_path, "automatic_llm_training")

    result = assess_guidance_compatibility(manifest, decision_path, date(2026, 11, 4))

    assert not result.is_valid
    failure = result.checks.loc[result.checks.check_id == "compatibility_selection_valid"].iloc[0]
    assert failure.status == "FAIL"
