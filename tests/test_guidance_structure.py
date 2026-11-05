import hashlib
from pathlib import Path

import pandas as pd

from scripts.generate_sample_ea_guidance import PAGES, build_pdf


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "data/unstructured/manifests/guidance_sources.csv"


def test_sample_guidance_manifest_matches_pdf() -> None:
    manifest = pd.read_csv(MANIFEST_PATH)
    row = manifest.loc[manifest["source_id"] == "EA-GUIDE-SAMPLE-001"].iloc[0]
    source_path = ROOT / row["source_path"]
    content = source_path.read_bytes()

    assert source_path.exists()
    assert content.startswith(b"%PDF-1.4")
    assert hashlib.sha256(content).hexdigest() == row["sha256"]
    assert not bool(row["is_synthetic"])
    assert row["review_status"] == "not_approved"


def test_sample_guidance_pdf_generation_is_deterministic() -> None:
    first = build_pdf(PAGES)
    second = build_pdf(PAGES)

    assert first == second
    assert b"Activation mode: Direct POC" in first
    assert b"GROWTH-G-001" in first
    assert b"SIMPLICITY-S-002" in first
