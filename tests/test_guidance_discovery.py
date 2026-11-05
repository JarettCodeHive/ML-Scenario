from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from cluster_c.unstructured.discovery import refresh_guidance_manifest
from cluster_c.unstructured.manifest import sha256_file


def test_discovery_builds_manifest_for_every_pdf(tmp_path: Path) -> None:
    project_root = tmp_path
    input_dir = project_root / "data/unstructured/input"
    input_dir.mkdir(parents=True)
    first = input_dir / "ea_growth_v2.pdf"
    second = input_dir / "customer-value.pdf"
    first.write_bytes(b"%PDF-1.4\nfirst")
    second.write_bytes(b"%PDF-1.4\nsecond")
    manifest_path = project_root / "data/unstructured/manifests/guidance_sources.csv"

    manifest = refresh_guidance_manifest(
        input_dir=input_dir,
        manifest_path=manifest_path,
        project_root=project_root,
        effective_date=date(2026, 11, 5),
    )

    assert len(manifest) == 2
    assert set(manifest.source_path) == {
        "data/unstructured/input/ea_growth_v2.pdf",
        "data/unstructured/input/customer-value.pdf",
    }
    growth = manifest.loc[manifest.source_path.str.endswith("ea_growth_v2.pdf")].iloc[0]
    assert growth.version == "v2"
    assert growth.sha256 == sha256_file(first)
    assert manifest.review_status.eq("not_approved").all()
    assert pd.read_csv(manifest_path).shape[0] == 2


def test_discovery_fails_clearly_when_input_has_no_pdf(tmp_path: Path) -> None:
    input_dir = tmp_path / "data/unstructured/input"
    input_dir.mkdir(parents=True)

    with pytest.raises(ValueError, match="No PDF files found"):
        refresh_guidance_manifest(
            input_dir=input_dir,
            manifest_path=tmp_path / "manifest.csv",
            project_root=tmp_path,
        )
