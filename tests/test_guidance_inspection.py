import json

import pytest

from cluster_c.unstructured.inspection import (
    GuidanceInspectionError,
    inspect_guidance_artifacts,
)
from cluster_c.unstructured.preparation import prepare_guidance, save_guidance_preparation


def _prepared_paths(tmp_path):
    paths = {
        "content": tmp_path / "guidance_content.parquet",
        "quality": tmp_path / "guidance_quality.csv",
        "rules": tmp_path / "guidance_rules.json",
        "compatibility": tmp_path / "guidance_compatibility_report.csv",
        "summary": tmp_path / "guidance_preparation_summary.json",
        "candidate_review": tmp_path / "guidance_candidate_review_report.csv",
    }
    result = prepare_guidance()
    save_guidance_preparation(
        result,
        paths["content"],
        paths["quality"],
        paths["rules"],
        paths["compatibility"],
        paths["summary"],
        paths["candidate_review"],
    )
    return paths


def _inspect(paths, source_id=None):
    return inspect_guidance_artifacts(
        source_id=source_id,
        content_path=paths["content"],
        quality_path=paths["quality"],
        rules_path=paths["rules"],
        compatibility_path=paths["compatibility"],
        summary_path=paths["summary"],
    )


def test_inspection_reports_extraction_and_candidate_details(tmp_path) -> None:
    paths = _prepared_paths(tmp_path)

    inspected = _inspect(paths)

    assert inspected.summary["extracted_page_count"] == 3
    assert inspected.sources.loc[0, "page_count"] == 3
    assert inspected.sources.loc[0, "text_character_count"] > 0
    assert len(inspected.candidates) == 6
    assert len(inspected.registry.active_rules) == 6
    assert inspected.failures.empty
    assert not inspected.warnings.empty
    assert inspected.pages["page_number"].tolist() == [1, 2, 3]


def test_inspection_can_be_restricted_to_a_registered_source(tmp_path) -> None:
    paths = _prepared_paths(tmp_path)
    source_id = _inspect(paths).sources.loc[0, "source_id"]

    inspected = _inspect(paths, source_id=source_id)

    assert inspected.sources["source_id"].tolist() == [source_id]
    assert inspected.pages["source_id"].eq(source_id).all()
    assert inspected.candidates["source_id"].eq(source_id).all()


def test_inspection_rejects_unknown_source(tmp_path) -> None:
    paths = _prepared_paths(tmp_path)

    with pytest.raises(GuidanceInspectionError, match="Unknown guidance source_id"):
        _inspect(paths, source_id="not-registered")


def test_inspection_rejects_inconsistent_summary(tmp_path) -> None:
    paths = _prepared_paths(tmp_path)
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    summary["candidate_rule_count"] += 1
    paths["summary"].write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(GuidanceInspectionError, match="artifact counts disagree"):
        _inspect(paths)


def test_inspection_requires_every_prepared_artifact(tmp_path) -> None:
    paths = _prepared_paths(tmp_path)
    paths["summary"].unlink()

    with pytest.raises(GuidanceInspectionError, match="Missing prepared guidance artifacts"):
        _inspect(paths)
