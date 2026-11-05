"""Safe onboarding of a new versioned EA guidance PDF."""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from cluster_c.config import (
    GUIDANCE_COMPATIBILITY_PATH,
    GUIDANCE_GOVERNANCE_PATH,
    GUIDANCE_MANIFEST_PATH,
    GUIDANCE_PDF_DIR,
    PROJECT_ROOT,
)

from .compatibility import ALLOWED_ONBOARDING_METHODS, COMPATIBILITY_COLUMNS
from .governance import ALLOWED_AUTHORITY_STATUSES, GOVERNANCE_COLUMNS
from .manifest import REQUIRED_MANIFEST_COLUMNS, sha256_file


SOURCE_ID_PATTERN = re.compile(r"^[A-Z][A-Z0-9_-]{2,99}$")
VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,49}$")


@dataclass(frozen=True)
class RegistrationResult:
    source_id: str
    destination: Path
    source_sha256: str
    manifest_record: dict[str, object]
    dry_run: bool


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")


def _read_registry(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path, dtype="string", keep_default_na=False)
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Registry {path} is missing columns: {missing}")
    return frame


def _write_registry_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def register_guidance_source(
    source_pdf: Path,
    *,
    source_id: str,
    title: str,
    version: str,
    effective_date: str | date,
    business_owner: str,
    technical_owner: str,
    authorized_rule_reviewer: str,
    backup_reviewer: str,
    source_repository: str,
    governance_contact: str,
    retention_policy: str,
    access_requirements: str,
    compatibility_assessor: str,
    selected_onboarding_method: str,
    assessment_notes: str,
    permitted_use_confirmed: bool,
    review_frequency_days: int = 365,
    supersedes_source_id: str = "",
    project_root: Path = PROJECT_ROOT,
    guidance_pdf_dir: Path = GUIDANCE_PDF_DIR,
    manifest_path: Path = GUIDANCE_MANIFEST_PATH,
    governance_path: Path = GUIDANCE_GOVERNANCE_PATH,
    compatibility_path: Path = GUIDANCE_COMPATIBILITY_PATH,
    registered_at: datetime | None = None,
    dry_run: bool = False,
) -> RegistrationResult:
    """Copy and register a real PDF without overwriting any existing version."""
    source_pdf = source_pdf.resolve()
    project_root = project_root.resolve()
    guidance_pdf_dir = guidance_pdf_dir.resolve()
    registered_at = registered_at or datetime.now(timezone.utc)

    if not guidance_pdf_dir.is_relative_to(project_root):
        raise ValueError("guidance_pdf_dir must remain inside project_root")

    if not source_pdf.is_file() or source_pdf.suffix.casefold() != ".pdf":
        raise ValueError(f"Source must be an existing PDF file: {source_pdf}")
    if not SOURCE_ID_PATTERN.fullmatch(source_id):
        raise ValueError("source_id must be an uppercase controlled identifier")
    if not VERSION_PATTERN.fullmatch(version):
        raise ValueError("version must contain only letters, numbers, dots, underscores, or hyphens")
    if not title.strip():
        raise ValueError("title must be populated")
    parsed_effective_date = pd.to_datetime(effective_date, errors="coerce")
    if pd.isna(parsed_effective_date):
        raise ValueError(f"effective_date is invalid: {effective_date!r}")
    required_people = {
        "business_owner": business_owner,
        "technical_owner": technical_owner,
        "authorized_rule_reviewer": authorized_rule_reviewer,
        "backup_reviewer": backup_reviewer,
        "source_repository": source_repository,
        "governance_contact": governance_contact,
        "retention_policy": retention_policy,
        "access_requirements": access_requirements,
        "compatibility_assessor": compatibility_assessor,
    }
    missing_details = [name for name, value in required_people.items() if not value.strip()]
    if missing_details:
        raise ValueError(f"Required onboarding details are missing: {missing_details}")
    if not permitted_use_confirmed:
        raise ValueError("Storage, extraction, rule derivation, and Streamlit reference must be confirmed")
    if selected_onboarding_method not in ALLOWED_ONBOARDING_METHODS:
        raise ValueError(f"Unsupported onboarding method: {selected_onboarding_method!r}")
    if review_frequency_days < 1:
        raise ValueError("review_frequency_days must be positive")

    manifest = _read_registry(manifest_path, REQUIRED_MANIFEST_COLUMNS)
    governance = _read_registry(governance_path, GOVERNANCE_COLUMNS)
    compatibility = _read_registry(compatibility_path, COMPATIBILITY_COLUMNS)
    if source_id in set(manifest["source_id"]):
        raise ValueError(f"source_id already exists: {source_id}")
    if source_id in set(governance["source_id"]) or source_id in set(compatibility["source_id"]):
        raise ValueError(f"source_id already exists in a supporting registry: {source_id}")
    duplicate_version = manifest["title"].str.casefold().eq(title.strip().casefold()) & manifest["version"].eq(version)
    if duplicate_version.any():
        raise ValueError(f"The same guidance title/version is already registered: {title!r} {version!r}")
    if supersedes_source_id and supersedes_source_id not in set(manifest["source_id"]):
        raise ValueError(f"supersedes_source_id is not registered: {supersedes_source_id}")

    filename = f"{_slug(source_id)}_{_slug(version)}.pdf"
    destination = guidance_pdf_dir / filename
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing guidance PDF: {destination}")
    digest = sha256_file(source_pdf)
    relative_destination = destination.relative_to(project_root).as_posix()
    manifest_record = {
        "source_id": source_id,
        "source_path": relative_destination,
        "source_type": "pdf",
        "guidance_type": "general_ea_guidance",
        "title": title.strip(),
        "version": version,
        "effective_date": parsed_effective_date.date().isoformat(),
        "review_status": "in_review",
        "is_synthetic": False,
        "sha256": digest,
    }
    governance_record = {
        "source_id": source_id,
        "business_owner": business_owner.strip(),
        "technical_owner": technical_owner.strip(),
        "authorized_rule_reviewer": authorized_rule_reviewer.strip(),
        "backup_reviewer": backup_reviewer.strip(),
        "source_repository": source_repository.strip(),
        "governance_contact": governance_contact.strip(),
        "authority_status": "in_review",
        "authority_confirmed_by": "",
        "authority_confirmed_at_utc": "",
        "storage_permitted": True,
        "text_extraction_permitted": True,
        "rule_derivation_permitted": True,
        "streamlit_reference_permitted": True,
        "retention_policy": retention_policy.strip(),
        "access_requirements": access_requirements.strip(),
        "retirement_date": "",
        "supersedes_source_id": supersedes_source_id,
        "review_frequency_days": review_frequency_days,
    }
    compatibility_record = {
        "source_id": source_id,
        "selected_onboarding_method": selected_onboarding_method,
        "assessed_by": compatibility_assessor.strip(),
        "assessed_at_utc": registered_at.isoformat(),
        "assessment_notes": assessment_notes.strip(),
    }
    if dry_run:
        return RegistrationResult(source_id, destination, digest, manifest_record, True)

    guidance_pdf_dir.mkdir(parents=True, exist_ok=True)
    registry_originals = {
        path: path.read_bytes() if path.exists() else None
        for path in (manifest_path, governance_path, compatibility_path)
    }
    try:
        with source_pdf.open("rb") as source, destination.open("xb") as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)
        _write_registry_atomic(
            pd.concat([manifest, pd.DataFrame([manifest_record])], ignore_index=True),
            manifest_path,
        )
        _write_registry_atomic(
            pd.concat([governance, pd.DataFrame([governance_record])], ignore_index=True),
            governance_path,
        )
        _write_registry_atomic(
            pd.concat([compatibility, pd.DataFrame([compatibility_record])], ignore_index=True),
            compatibility_path,
        )
    except Exception:
        if destination.exists():
            destination.unlink()
        for path, original in registry_originals.items():
            if original is None:
                if path.exists():
                    path.unlink()
            else:
                path.write_bytes(original)
        raise
    return RegistrationResult(source_id, destination, digest, manifest_record, False)
