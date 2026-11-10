from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from cluster_c.unstructured.compatibility import COMPATIBILITY_COLUMNS
from cluster_c.unstructured.governance import GOVERNANCE_COLUMNS
from cluster_c.unstructured.manifest import REQUIRED_MANIFEST_COLUMNS, sha256_file
from cluster_c.unstructured.registration import register_guidance_source
from scripts.generate_sample_ea_guidance import PAGES, build_pdf


def registries(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    project_root = tmp_path / "project"
    pdf_dir = project_root / "data/unstructured/input/guidance/pdf"
    manifest_path = project_root / "data/unstructured/manifests/guidance_sources.csv"
    governance_path = project_root / "data/unstructured/manifests/guidance_governance.csv"
    compatibility_path = project_root / "data/unstructured/manifests/guidance_compatibility.csv"
    manifest_path.parent.mkdir(parents=True)
    pd.DataFrame(columns=REQUIRED_MANIFEST_COLUMNS).to_csv(manifest_path, index=False)
    pd.DataFrame(columns=GOVERNANCE_COLUMNS).to_csv(governance_path, index=False)
    pd.DataFrame(columns=COMPATIBILITY_COLUMNS).to_csv(compatibility_path, index=False)
    return project_root, pdf_dir, manifest_path, governance_path, compatibility_path


def source_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "authoritative-guidance.pdf"
    path.write_bytes(build_pdf(PAGES))
    return path


def register(tmp_path: Path, **overrides):
    project_root, pdf_dir, manifest_path, governance_path, compatibility_path = registries(tmp_path)
    values = {
        "source_pdf": source_pdf(tmp_path),
        "source_id": "EA-GUIDE-REAL-001",
        "title": "EA Term Guidance",
        "version": "v1",
        "effective_date": "2026-12-01",
        "business_owner": "EA Business Owner",
        "technical_owner": "EA Model Owner",
        "authorized_rule_reviewer": "EA Rule Reviewer",
        "backup_reviewer": "EA Backup Reviewer",
        "source_repository": "Approved EA Repository",
        "governance_contact": "ea-governance@example.test",
        "retention_policy": "Approved EA retention policy",
        "access_requirements": "Authorized EA users only",
        "compatibility_assessor": "EA Compatibility Reviewer",
        "selected_onboarding_method": "direct_controlled_pdf",
        "assessment_notes": "Controlled rule format confirmed",
        "permitted_use_confirmed": True,
        "project_root": project_root,
        "guidance_pdf_dir": pdf_dir,
        "manifest_path": manifest_path,
        "governance_path": governance_path,
        "compatibility_path": compatibility_path,
        "registered_at": datetime(2026, 11, 4, 12, 0, tzinfo=timezone.utc),
    }
    values.update(overrides)
    result = register_guidance_source(**values)
    return result, values


def test_register_real_pdf_and_all_review_records(tmp_path: Path) -> None:
    result, values = register(tmp_path)

    assert result.destination.is_file()
    assert result.destination.name == "ea-guide-real-001_v1.pdf"
    assert result.source_sha256 == sha256_file(result.destination)
    manifest = pd.read_csv(values["manifest_path"])
    governance = pd.read_csv(values["governance_path"])
    compatibility = pd.read_csv(values["compatibility_path"])
    assert manifest.loc[0, "review_status"] == "in_review"
    assert not bool(manifest.loc[0, "is_synthetic"])
    assert manifest.loc[0, "sha256"] == result.source_sha256
    assert governance.loc[0, "authority_status"] == "in_review"
    assert bool(governance.loc[0, "storage_permitted"])
    assert compatibility.loc[0, "selected_onboarding_method"] == "direct_controlled_pdf"
    assert compatibility.loc[0, "assessed_by"] == "EA Compatibility Reviewer"


def test_registration_refuses_duplicate_source_id_without_overwriting(tmp_path: Path) -> None:
    first, values = register(tmp_path)

    with pytest.raises(ValueError, match="source_id already exists"):
        register_guidance_source(**values)

    assert first.destination.is_file()
    assert len(pd.read_csv(values["manifest_path"])) == 1


def test_registration_requires_explicit_permitted_use_confirmation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be confirmed"):
        register(tmp_path, permitted_use_confirmed=False)


def test_dry_run_does_not_copy_or_change_registries(tmp_path: Path) -> None:
    result, values = register(tmp_path, dry_run=True)

    assert result.dry_run
    assert not result.destination.exists()
    assert pd.read_csv(values["manifest_path"]).empty
    assert pd.read_csv(values["governance_path"]).empty
    assert pd.read_csv(values["compatibility_path"]).empty


def test_new_version_preserves_previous_file_and_record(tmp_path: Path) -> None:
    first, values = register(tmp_path)
    second_pdf = tmp_path / "authoritative-guidance-v2.pdf"
    second_pdf.write_bytes(build_pdf(PAGES + [["Version two appendix"]]))

    second = register_guidance_source(
        second_pdf,
        **{
            **{key: value for key, value in values.items() if key != "source_pdf"},
            "source_id": "EA-GUIDE-REAL-002",
            "version": "v2",
            "supersedes_source_id": "EA-GUIDE-REAL-001",
        },
    )

    assert first.destination.is_file()
    assert second.destination.is_file()
    assert first.destination != second.destination
    manifest = pd.read_csv(values["manifest_path"])
    governance = pd.read_csv(values["governance_path"])
    assert manifest["source_id"].tolist() == ["EA-GUIDE-REAL-001", "EA-GUIDE-REAL-002"]
    assert governance.loc[1, "supersedes_source_id"] == "EA-GUIDE-REAL-001"
