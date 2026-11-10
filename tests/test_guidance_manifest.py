from pathlib import Path

import pandas as pd

from cluster_c.config import GUIDANCE_MANIFEST_PATH, GUIDANCE_PDF_DIR, PROJECT_ROOT
from cluster_c.unstructured.manifest import validate_guidance_manifest


def test_repository_guidance_manifest_is_valid() -> None:
    result = validate_guidance_manifest()

    assert result.is_valid
    assert result.manifest["source_id"].tolist() == ["EA-GUIDE-SAMPLE-001"]
    assert result.manifest["is_synthetic"].tolist() == [False]
    assert result.checks["status"].eq("PASS").all()


def test_manifest_hash_mismatch_is_blocking(tmp_path: Path) -> None:
    manifest = pd.read_csv(GUIDANCE_MANIFEST_PATH, dtype="string", keep_default_na=False)
    manifest.loc[0, "sha256"] = "0" * 64
    manifest_path = tmp_path / "guidance_sources.csv"
    manifest.to_csv(manifest_path, index=False)

    result = validate_guidance_manifest(manifest_path)

    assert not result.is_valid
    failed = result.checks.loc[result.checks.status == "FAIL"]
    assert "source_hash_matches" in set(failed["check_id"])


def test_manifest_rejects_source_outside_guidance_directory(tmp_path: Path) -> None:
    manifest = pd.read_csv(GUIDANCE_MANIFEST_PATH, dtype="string", keep_default_na=False)
    manifest.loc[0, "source_path"] = "requirements.txt"
    manifest_path = tmp_path / "guidance_sources.csv"
    manifest.to_csv(manifest_path, index=False)

    result = validate_guidance_manifest(
        manifest_path=manifest_path,
        project_root=PROJECT_ROOT,
        guidance_pdf_dir=GUIDANCE_PDF_DIR,
    )

    assert not result.is_valid
    failed = result.checks.loc[result.checks.status == "FAIL"]
    assert "source_path_allowed" in set(failed["check_id"])


def test_manifest_reports_missing_required_columns(tmp_path: Path) -> None:
    manifest = pd.read_csv(GUIDANCE_MANIFEST_PATH).drop(columns="review_status")
    manifest_path = tmp_path / "guidance_sources.csv"
    manifest.to_csv(manifest_path, index=False)

    result = validate_guidance_manifest(manifest_path)

    assert not result.is_valid
    check = result.checks.loc[result.checks.check_id == "manifest_required_columns"].iloc[0]
    assert check.status == "FAIL"
    assert "review_status" in check.detail
