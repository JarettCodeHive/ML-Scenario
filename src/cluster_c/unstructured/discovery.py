"""Automatic PDF discovery and manifest generation for the POC workflow."""

from __future__ import annotations

import os
import re
from datetime import date
from pathlib import Path

import pandas as pd

from cluster_c.config import (
    GUIDANCE_MANIFEST_PATH,
    GUIDANCE_PDF_DIR,
    PROJECT_ROOT,
)

from .manifest import REQUIRED_MANIFEST_COLUMNS, sha256_file


def _slug(value: str) -> str:
    normalized = re.sub(r"[^A-Z0-9]+", "-", value.upper()).strip("-")
    return normalized or "PDF"


def _version(path: Path) -> str:
    match = re.search(r"(?:^|[_-])(v\d+(?:\.\d+)*)", path.stem, re.IGNORECASE)
    return match.group(1).lower() if match else "v1"


def _existing_records(manifest_path: Path) -> dict[str, dict[str, str]]:
    if not manifest_path.is_file():
        return {}
    try:
        frame = pd.read_csv(manifest_path, dtype="string", keep_default_na=False)
    except (OSError, UnicodeError, pd.errors.ParserError):
        return {}
    if not set(REQUIRED_MANIFEST_COLUMNS).issubset(frame.columns):
        return {}
    return {
        str(row["source_path"]): {column: str(row[column]) for column in REQUIRED_MANIFEST_COLUMNS}
        for _, row in frame.iterrows()
    }


def refresh_guidance_manifest(
    input_dir: Path = GUIDANCE_PDF_DIR,
    manifest_path: Path = GUIDANCE_MANIFEST_PATH,
    project_root: Path = PROJECT_ROOT,
    effective_date: date | None = None,
) -> pd.DataFrame:
    """Discover every PDF and atomically generate its source manifest."""
    input_dir = input_dir.resolve()
    project_root = project_root.resolve()
    effective_date = effective_date or date.today()
    pdf_paths = sorted(
        (path for path in input_dir.rglob("*.pdf") if path.is_file()),
        key=lambda path: path.relative_to(input_dir).as_posix().casefold(),
    )
    if not pdf_paths:
        raise ValueError(f"No PDF files found under {input_dir}")

    existing = _existing_records(manifest_path)
    rows: list[dict[str, str]] = []
    used_ids: set[str] = set()
    for path in pdf_paths:
        relative_path = path.relative_to(project_root).as_posix()
        previous = existing.get(relative_path, {})
        digest = sha256_file(path)
        proposed_id = previous.get("source_id") or f"EA-GUIDE-{_slug(path.stem)}"
        source_id = proposed_id
        if source_id in used_ids:
            source_id = f"{proposed_id}-{digest[:8].upper()}"
        used_ids.add(source_id)
        rows.append({
            "source_id": source_id,
            "source_path": relative_path,
            "source_type": "pdf",
            "guidance_type": "general_ea_guidance",
            "title": previous.get("title") or path.stem.replace("_", " ").replace("-", " ").title(),
            "version": previous.get("version") or _version(path),
            "effective_date": previous.get("effective_date") or effective_date.isoformat(),
            # These legacy fields remain only for schema compatibility. Direct
            # POC activation does not consult either value.
            "review_status": "not_approved",
            "is_synthetic": previous.get("is_synthetic") or "false",
            "sha256": digest,
        })

    manifest = pd.DataFrame(rows, columns=REQUIRED_MANIFEST_COLUMNS)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_name(f".{manifest_path.name}.tmp")
    manifest.to_csv(temporary, index=False)
    os.replace(temporary, manifest_path)
    return manifest
