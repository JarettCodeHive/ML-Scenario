"""Manifest loading and validation for general EA guidance sources."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from cluster_c.config import GUIDANCE_MANIFEST_PATH, GUIDANCE_PDF_DIR, PROJECT_ROOT


REQUIRED_MANIFEST_COLUMNS = [
    "source_id",
    "source_path",
    "source_type",
    "guidance_type",
    "title",
    "version",
    "effective_date",
    "review_status",
    "is_synthetic",
    "sha256",
]
ALLOWED_SOURCE_TYPES = {"pdf"}
ALLOWED_GUIDANCE_TYPES = {"general_ea_guidance"}
ALLOWED_REVIEW_STATUSES = {
    "not_approved",
    "in_review",
    "approved",
    "retired",
    "rejected",
}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
QUALITY_COLUMNS = ["check_id", "source_id", "status", "severity", "description", "detail"]


@dataclass(frozen=True)
class ManifestValidationResult:
    manifest: pd.DataFrame
    checks: pd.DataFrame

    @property
    def is_valid(self) -> bool:
        return not self.checks["status"].eq("FAIL").any()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_boolean(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def validate_guidance_manifest(
    manifest_path: Path = GUIDANCE_MANIFEST_PATH,
    project_root: Path = PROJECT_ROOT,
    guidance_pdf_dir: Path = GUIDANCE_PDF_DIR,
) -> ManifestValidationResult:
    """Load and validate the guidance inventory without processing PDF text."""
    checks: list[dict[str, str]] = []

    def add(
        check_id: str,
        passed: bool,
        description: str,
        detail: str = "",
        source_id: str = "<MANIFEST>",
        severity: str = "blocking",
    ) -> None:
        checks.append({
            "check_id": check_id,
            "source_id": source_id,
            "status": "PASS" if passed else "FAIL",
            "severity": severity,
            "description": description,
            "detail": detail,
        })

    try:
        manifest = pd.read_csv(manifest_path, dtype="string", keep_default_na=False)
    except (OSError, pd.errors.ParserError, UnicodeError) as exc:
        add("manifest_readable", False, "Guidance manifest must be a readable CSV", str(exc))
        return ManifestValidationResult(
            pd.DataFrame(columns=REQUIRED_MANIFEST_COLUMNS),
            pd.DataFrame(checks, columns=QUALITY_COLUMNS),
        )

    add("manifest_readable", True, "Guidance manifest must be a readable CSV")
    missing_columns = sorted(set(REQUIRED_MANIFEST_COLUMNS) - set(manifest.columns))
    add(
        "manifest_required_columns",
        not missing_columns,
        "All required guidance manifest columns must exist",
        f"missing={missing_columns}" if missing_columns else "",
    )
    if missing_columns:
        return ManifestValidationResult(manifest, pd.DataFrame(checks, columns=QUALITY_COLUMNS))

    add("manifest_not_empty", not manifest.empty, "Guidance manifest must contain at least one source")
    if manifest.empty:
        return ManifestValidationResult(manifest, pd.DataFrame(checks, columns=QUALITY_COLUMNS))

    source_ids = manifest["source_id"].str.strip()
    duplicate_ids = sorted(source_ids[source_ids.duplicated(keep=False) & source_ids.ne("")].unique())
    add(
        "source_id_present_unique",
        source_ids.ne("").all() and not duplicate_ids,
        "Source IDs must be populated and unique",
        f"duplicates={duplicate_ids}" if duplicate_ids else "",
    )

    normalized = manifest.copy()
    project_root = project_root.resolve()
    guidance_pdf_dir = guidance_pdf_dir.resolve()
    resolved_paths: list[str | None] = []
    parsed_dates: list[pd.Timestamp | pd.NaT] = []
    parsed_synthetic: list[bool | None] = []

    for index, row in manifest.iterrows():
        source_id = str(row["source_id"]).strip() or f"<ROW:{index + 2}>"
        source_type = str(row["source_type"]).strip().lower()
        guidance_type = str(row["guidance_type"]).strip().lower()
        version = str(row["version"]).strip()
        title = str(row["title"]).strip()
        review_status = str(row["review_status"]).strip().lower()
        raw_path = str(row["source_path"]).strip()
        raw_hash = str(row["sha256"]).strip().lower()
        effective_date = pd.to_datetime(row["effective_date"], errors="coerce")
        synthetic = _parse_boolean(row["is_synthetic"])

        add(
            "source_type_allowed",
            source_type in ALLOWED_SOURCE_TYPES,
            "Guidance source type must be supported",
            f"received={source_type!r}",
            source_id,
        )
        add(
            "guidance_type_allowed",
            guidance_type in ALLOWED_GUIDANCE_TYPES,
            "Guidance type must be supported",
            f"received={guidance_type!r}",
            source_id,
        )
        add("source_title_present", bool(title), "Source title must be populated", source_id=source_id)
        add("source_version_present", bool(version), "Source version must be populated", source_id=source_id)
        add(
            "effective_date_valid",
            not pd.isna(effective_date),
            "Effective date must be parseable",
            f"received={row['effective_date']!r}",
            source_id,
        )
        add(
            "review_status_allowed",
            review_status in ALLOWED_REVIEW_STATUSES,
            "Review status must use the controlled vocabulary",
            f"received={review_status!r}",
            source_id,
        )
        add(
            "synthetic_flag_valid",
            synthetic is not None,
            "Synthetic flag must be an explicit boolean",
            f"received={row['is_synthetic']!r}",
            source_id,
        )
        add(
            "sha256_format_valid",
            bool(SHA256_PATTERN.fullmatch(raw_hash)),
            "SHA-256 must contain 64 lowercase hexadecimal characters",
            source_id=source_id,
        )

        source_path = Path(raw_path)
        path_is_relative = bool(raw_path) and not source_path.is_absolute()
        add(
            "source_path_relative",
            path_is_relative,
            "Source path must be populated and repository-relative",
            f"received={raw_path!r}",
            source_id,
        )
        resolved_path = (project_root / source_path).resolve() if path_is_relative else None
        inside_pdf_dir = resolved_path is not None and resolved_path.is_relative_to(guidance_pdf_dir)
        add(
            "source_path_allowed",
            inside_pdf_dir,
            "Source path must remain inside the approved guidance PDF directory",
            source_id=source_id,
        )
        exists = bool(inside_pdf_dir and resolved_path and resolved_path.is_file())
        add("source_file_present", exists, "Guidance source file must exist", source_id=source_id)

        hash_matches = False
        hash_detail = ""
        if exists and SHA256_PATTERN.fullmatch(raw_hash):
            actual_hash = sha256_file(resolved_path)
            hash_matches = actual_hash == raw_hash
            if not hash_matches:
                hash_detail = f"expected={raw_hash}; actual={actual_hash}"
        add(
            "source_hash_matches",
            hash_matches,
            "Source file must match its reviewed SHA-256",
            hash_detail,
            source_id,
        )

        resolved_paths.append(str(resolved_path) if inside_pdf_dir and resolved_path else None)
        parsed_dates.append(effective_date)
        parsed_synthetic.append(synthetic)

    normalized["source_id"] = source_ids
    normalized["source_type"] = normalized["source_type"].str.strip().str.lower()
    normalized["guidance_type"] = normalized["guidance_type"].str.strip().str.lower()
    normalized["title"] = normalized["title"].str.strip()
    normalized["version"] = normalized["version"].str.strip()
    normalized["review_status"] = normalized["review_status"].str.strip().str.lower()
    normalized["sha256"] = normalized["sha256"].str.strip().str.lower()
    normalized["effective_date"] = pd.to_datetime(parsed_dates)
    normalized["is_synthetic"] = pd.array(parsed_synthetic, dtype="boolean")
    normalized["resolved_path"] = resolved_paths
    return ManifestValidationResult(normalized, pd.DataFrame(checks, columns=QUALITY_COLUMNS))
