from cluster_c.config import (
    DATA_DIR,
    GUIDANCE_APPROVALS_PATH,
    GUIDANCE_CANDIDATE_REVIEW_REPORT_PATH,
    GUIDANCE_CANDIDATE_REVIEWS_PATH,
    GUIDANCE_CACHE_DIR,
    GUIDANCE_CONTENT_PATH,
    GUIDANCE_COMPATIBILITY_PATH,
    GUIDANCE_COMPATIBILITY_REPORT_PATH,
    GUIDANCE_GOVERNANCE_PATH,
    GUIDANCE_MANIFEST_PATH,
    GUIDANCE_PDF_DIR,
    GUIDANCE_PREPARED_DIR,
    GUIDANCE_QUALITY_PATH,
    GUIDANCE_REVIEW_POLICY_PATH,
    GUIDANCE_RELEASE_METADATA_PATH,
    GUIDANCE_RULES_PATH,
    GUIDANCE_SUMMARY_PATH,
    GUIDANCE_SOURCE_APPROVALS_PATH,
    PROJECT_ROOT,
    UNSTRUCTURED_DIR,
)


def test_guidance_paths_are_repository_relative_and_separate_from_structured_data() -> None:
    assert UNSTRUCTURED_DIR == DATA_DIR / "unstructured"
    assert GUIDANCE_PDF_DIR == UNSTRUCTURED_DIR / "input" / "guidance" / "pdf"
    assert GUIDANCE_MANIFEST_PATH == UNSTRUCTURED_DIR / "manifests" / "guidance_sources.csv"
    assert GUIDANCE_APPROVALS_PATH == UNSTRUCTURED_DIR / "manifests" / "guidance_rule_approvals.csv"
    assert GUIDANCE_SOURCE_APPROVALS_PATH == UNSTRUCTURED_DIR / "manifests" / "guidance_source_approvals.csv"
    assert GUIDANCE_CANDIDATE_REVIEWS_PATH == UNSTRUCTURED_DIR / "manifests" / "guidance_candidate_reviews.csv"
    assert GUIDANCE_REVIEW_POLICY_PATH == UNSTRUCTURED_DIR / "manifests" / "guidance_review_policy.csv"
    assert GUIDANCE_GOVERNANCE_PATH == UNSTRUCTURED_DIR / "manifests" / "guidance_governance.csv"
    assert GUIDANCE_COMPATIBILITY_PATH == UNSTRUCTURED_DIR / "manifests" / "guidance_compatibility.csv"
    assert GUIDANCE_PREPARED_DIR == UNSTRUCTURED_DIR / "prepared"
    assert GUIDANCE_CONTENT_PATH == GUIDANCE_PREPARED_DIR / "guidance_content.parquet"
    assert GUIDANCE_RULES_PATH == GUIDANCE_PREPARED_DIR / "guidance_rules.json"
    assert GUIDANCE_QUALITY_PATH == GUIDANCE_PREPARED_DIR / "guidance_quality.csv"
    assert GUIDANCE_COMPATIBILITY_REPORT_PATH == GUIDANCE_PREPARED_DIR / "guidance_compatibility_report.csv"
    assert GUIDANCE_SUMMARY_PATH == GUIDANCE_PREPARED_DIR / "guidance_preparation_summary.json"
    assert GUIDANCE_CANDIDATE_REVIEW_REPORT_PATH == GUIDANCE_PREPARED_DIR / "guidance_candidate_review_report.csv"
    assert GUIDANCE_RELEASE_METADATA_PATH == GUIDANCE_PREPARED_DIR / "guidance_release_metadata.json"
    assert GUIDANCE_CACHE_DIR == UNSTRUCTURED_DIR / "cache"
    assert all(
        path.is_relative_to(PROJECT_ROOT)
        for path in (
            GUIDANCE_PDF_DIR,
            GUIDANCE_MANIFEST_PATH,
            GUIDANCE_APPROVALS_PATH,
            GUIDANCE_SOURCE_APPROVALS_PATH,
            GUIDANCE_CANDIDATE_REVIEWS_PATH,
            GUIDANCE_REVIEW_POLICY_PATH,
            GUIDANCE_GOVERNANCE_PATH,
            GUIDANCE_COMPATIBILITY_PATH,
            GUIDANCE_CONTENT_PATH,
            GUIDANCE_RULES_PATH,
            GUIDANCE_QUALITY_PATH,
            GUIDANCE_COMPATIBILITY_REPORT_PATH,
            GUIDANCE_SUMMARY_PATH,
            GUIDANCE_CANDIDATE_REVIEW_REPORT_PATH,
            GUIDANCE_RELEASE_METADATA_PATH,
            GUIDANCE_CACHE_DIR,
        )
    )


def test_required_guidance_input_locations_exist() -> None:
    assert GUIDANCE_PDF_DIR.is_dir()
    assert GUIDANCE_MANIFEST_PATH.is_file()
    assert GUIDANCE_APPROVALS_PATH.is_file()
    assert GUIDANCE_SOURCE_APPROVALS_PATH.is_file()
    assert GUIDANCE_CANDIDATE_REVIEWS_PATH.is_file()
    assert GUIDANCE_REVIEW_POLICY_PATH.is_file()
    assert GUIDANCE_GOVERNANCE_PATH.is_file()
    assert GUIDANCE_COMPATIBILITY_PATH.is_file()
    assert GUIDANCE_PREPARED_DIR.is_dir()
    assert GUIDANCE_CACHE_DIR.is_dir()
