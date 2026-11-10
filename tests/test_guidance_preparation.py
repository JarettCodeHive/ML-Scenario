import json
from datetime import datetime, timezone

import pandas as pd

from cluster_c.unstructured.preparation import prepare_guidance, save_guidance_preparation


def test_guidance_preparation_extracts_page_provenance() -> None:
    extracted_at = datetime(2026, 11, 4, 12, 0, tzinfo=timezone.utc)

    result = prepare_guidance(extracted_at=extracted_at)

    assert result.is_valid
    assert result.summary["extracted_source_count"] == 1
    assert result.summary["extracted_page_count"] == 3
    assert result.summary["candidate_rule_count"] == 6
    assert result.summary["active_rule_count"] == 6
    assert result.summary["activation_mode"] == "direct_poc"
    assert result.summary["quality_warning_count"] >= 1
    assert result.content["page_number"].tolist() == [1, 2, 3]
    assert result.content["source_id"].nunique() == 1
    assert result.content["source_sha256"].str.len().eq(64).all()
    assert result.content["review_status"].eq("not_approved").all()
    assert result.content["extracted_at_utc"].eq(extracted_at).all()
    assert "GROWTH-G-001" in result.content.loc[result.content.page_number == 2, "page_text"].iloc[0]


def test_guidance_preparation_writes_parquet_and_quality_csv(tmp_path) -> None:
    result = prepare_guidance()
    content_path = tmp_path / "guidance_content.parquet"
    quality_path = tmp_path / "guidance_quality.csv"
    rules_path = tmp_path / "guidance_rules.json"
    compatibility_path = tmp_path / "guidance_compatibility_report.csv"
    summary_path = tmp_path / "guidance_preparation_summary.json"
    candidate_review_path = tmp_path / "guidance_candidate_review_report.csv"

    paths = save_guidance_preparation(
        result,
        content_path,
        quality_path,
        rules_path,
        compatibility_path,
        summary_path,
        candidate_review_path,
    )
    restored_content = pd.read_parquet(paths["content"], engine="pyarrow")
    restored_quality = pd.read_csv(paths["quality"])
    restored_rules = json.loads(paths["rules"].read_text(encoding="utf-8"))
    restored_compatibility = pd.read_csv(paths["compatibility"])
    restored_summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    restored_candidate_review = pd.read_csv(paths["candidate_review"])

    assert len(restored_content) == 3
    assert restored_content.columns.tolist() == result.content.columns.tolist()
    assert not restored_quality.status.eq("FAIL").any()
    assert set(restored_quality.severity) == {"blocking", "review_required", "information"}
    assert restored_rules["candidate_rule_count"] == 6
    assert restored_rules["active_rule_count"] == 6
    assert set(restored_rules["active_rule_ids"]) == {
        "GROWTH-G-001",
        "GROWTH-G-002",
        "CUSTOMER-CV-001",
        "CUSTOMER-CV-002",
        "SIMPLICITY-S-001",
        "SIMPLICITY-S-002",
    }
    assert restored_compatibility.loc[0, "format_compatibility"] == "controlled_rules"
    assert bool(restored_compatibility.loc[0, "activation_compatible"])
    assert restored_summary["generated_at_utc"] == result.generated_at_utc.isoformat()
    assert restored_summary["extracted_page_count"] == len(restored_content)
    assert restored_summary["candidate_rule_count"] == restored_rules["candidate_rule_count"]
    assert restored_candidate_review.empty
