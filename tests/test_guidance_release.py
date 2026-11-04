import json
import sys
from datetime import datetime, timezone

import pytest

from scripts.build_ea_guidance_release import parse_expected_rule_ids
from cluster_c.unstructured.release import (
    GuidanceReleaseError,
    build_guidance_release,
    publish_prepared_directory,
)


DEPLOYMENT_TIME = datetime(2026, 11, 4, 15, 0, tzinfo=timezone.utc)


def _metadata_files(tmp_path):
    model = tmp_path / "model_metadata.json"
    preparation = tmp_path / "preparation_summary.json"
    model.write_text(json.dumps({
        "model_version": "test-model-v1",
        "source_dataset_version": "test-dataset-v1",
        "dataset_sha256": "d" * 64,
    }), encoding="utf-8")
    preparation.write_text(json.dumps({
        "source_dataset_version": "test-dataset-v1",
    }), encoding="utf-8")
    return model, preparation


def test_release_build_publishes_validated_bundle_and_metadata(tmp_path) -> None:
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()
    (prepared_dir / ".gitkeep").touch()
    (prepared_dir / "old-generated.txt").write_text("old", encoding="utf-8")
    model, preparation = _metadata_files(tmp_path)

    metadata = build_guidance_release(
        application_version="test-app-v1",
        expected_active_rule_ids=set(),
        warning_policy="report",
        prepared_dir=prepared_dir,
        model_metadata_path=model,
        preparation_summary_path=preparation,
        deployment_time=DEPLOYMENT_TIME,
    )

    assert not (prepared_dir / "old-generated.txt").exists()
    assert (prepared_dir / ".gitkeep").is_file()
    for filename in (
        "guidance_content.parquet",
        "guidance_rules.json",
        "guidance_quality.csv",
        "guidance_compatibility_report.csv",
        "guidance_preparation_summary.json",
        "guidance_candidate_review_report.csv",
        "guidance_release_metadata.json",
    ):
        assert (prepared_dir / filename).is_file()
    restored = json.loads((prepared_dir / "guidance_release_metadata.json").read_text(encoding="utf-8"))
    assert restored == metadata
    assert metadata["application_version"] == "test-app-v1"
    assert metadata["structured_model_version"] == "test-model-v1"
    assert metadata["structured_dataset_version"] == "test-dataset-v1"
    assert metadata["active_rule_ids"] == []
    assert metadata["active_rule_count"] == 0
    assert metadata["quality_failure_count"] == 0
    assert metadata["guidance_sources"][0]["source_id"] == "EA-GUIDE-SAMPLE-001"
    assert metadata["guidance_sources"][0]["active"] is False
    assert metadata["deployment_time_utc"] == DEPLOYMENT_TIME.isoformat()


def test_active_rule_mismatch_does_not_replace_previous_release(tmp_path) -> None:
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()
    marker = prepared_dir / "previous-release.txt"
    marker.write_text("keep", encoding="utf-8")
    model, preparation = _metadata_files(tmp_path)

    with pytest.raises(GuidanceReleaseError, match="Active rule IDs do not match"):
        build_guidance_release(
            application_version="test-app-v2",
            expected_active_rule_ids={"NOT-ACTIVE"},
            warning_policy="report",
            prepared_dir=prepared_dir,
            model_metadata_path=model,
            preparation_summary_path=preparation,
            deployment_time=DEPLOYMENT_TIME,
        )

    assert marker.read_text(encoding="utf-8") == "keep"


def test_directory_publication_restores_previous_release_on_failure(tmp_path) -> None:
    target = tmp_path / "prepared"
    staged = tmp_path / "staged"
    target.mkdir()
    staged.mkdir()
    (target / "release.txt").write_text("previous", encoding="utf-8")
    (staged / "release.txt").write_text("new", encoding="utf-8")

    def fail_after_backup():
        raise RuntimeError("simulated activation failure")

    with pytest.raises(RuntimeError, match="simulated"):
        publish_prepared_directory(staged, target, after_backup=fail_after_backup)

    assert (target / "release.txt").read_text(encoding="utf-8") == "previous"
    assert (staged / "release.txt").read_text(encoding="utf-8") == "new"


def test_expected_rule_ids_require_explicit_none_or_values() -> None:
    assert parse_expected_rule_ids("none") == set()
    assert parse_expected_rule_ids("RULE-A,RULE-B") == {"RULE-A", "RULE-B"}
    with pytest.raises(Exception, match="explicit value 'none'"):
        parse_expected_rule_ids(" , ")


def test_deployment_never_starts_streamlit_when_release_gate_fails(monkeypatch) -> None:
    import scripts.deploy_streamlit as deploy

    started = False

    def fail_release(**kwargs):
        raise GuidanceReleaseError("release gate failed")

    def start_process(*args, **kwargs):
        nonlocal started
        started = True

    monkeypatch.setattr(deploy, "build_guidance_release", fail_release)
    monkeypatch.setattr(deploy.subprocess, "run", start_process)
    monkeypatch.setattr(sys, "argv", [
        "deploy_streamlit.py",
        "--application-version", "test",
        "--expected-active-rule-ids", "none",
    ])

    with pytest.raises(GuidanceReleaseError, match="release gate failed"):
        deploy.main()
    assert started is False
