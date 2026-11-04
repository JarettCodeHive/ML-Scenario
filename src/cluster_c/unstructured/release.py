"""Build, validate, and atomically publish EA guidance release artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from tempfile import mkdtemp
from typing import Callable

import pandas as pd

from cluster_c.config import (
    GUIDANCE_APPROVALS_PATH,
    GUIDANCE_PREPARED_DIR,
    GUIDANCE_RELEASE_METADATA_PATH,
    GUIDANCE_SOURCE_APPROVALS_PATH,
    MODEL_DIR,
    PREPARED_DIR,
)

from .decision_rules import load_guidance_registry
from .preparation import prepare_guidance, save_guidance_preparation


class GuidanceReleaseError(ValueError):
    """A guidance release failed validation or publication."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GuidanceReleaseError(f"Unable to read {label} metadata {path}: {exc}") from exc


def _atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def publish_prepared_directory(
    staged_dir: Path,
    target_dir: Path,
    *,
    after_backup: Callable[[], None] | None = None,
) -> None:
    """Replace the generated directory and restore the previous one on failure."""
    if not staged_dir.is_dir():
        raise GuidanceReleaseError(f"Staged guidance directory does not exist: {staged_dir}")
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    backup_dir = target_dir.with_name(f".{target_dir.name}.rollback-{os.getpid()}")
    if backup_dir.exists():
        raise GuidanceReleaseError(f"Rollback directory already exists: {backup_dir}")
    previous_exists = target_dir.exists()
    try:
        if previous_exists:
            target_dir.replace(backup_dir)
        if after_backup:
            after_backup()
        staged_dir.replace(target_dir)
    except Exception:
        if target_dir.exists() and not previous_exists:
            shutil.rmtree(target_dir)
        if backup_dir.exists() and not target_dir.exists():
            backup_dir.replace(target_dir)
        raise
    else:
        if backup_dir.exists():
            shutil.rmtree(backup_dir)


def _approval_identities(active_rules, rule_approvals_path: Path, source_approvals_path: Path) -> dict[str, list[str]]:
    active_ids = {rule.rule_id for rule in active_rules}
    active_sources = {rule.source_id for rule in active_rules}
    rule_approvals = pd.read_csv(rule_approvals_path, dtype="string", keep_default_na=False)
    source_approvals = pd.read_csv(source_approvals_path, dtype="string", keep_default_na=False)
    rule_reviewers = sorted(set(
        rule_approvals.loc[
            rule_approvals.rule_id.isin(active_ids) & rule_approvals.review_status.eq("approved"),
            "reviewed_by",
        ].loc[lambda values: values.ne("")]
    ))
    source_reviewers = sorted(set(
        source_approvals.loc[
            source_approvals.source_id.isin(active_sources) & source_approvals.review_status.eq("approved"),
            "reviewed_by",
        ].loc[lambda values: values.ne("")]
    ))
    return {"rule_reviewers": rule_reviewers, "source_reviewers": source_reviewers}


def build_guidance_release(
    *,
    application_version: str,
    expected_active_rule_ids: set[str],
    warning_policy: str = "fail",
    prepared_dir: Path = GUIDANCE_PREPARED_DIR,
    model_metadata_path: Path = MODEL_DIR / "metadata.json",
    preparation_summary_path: Path = PREPARED_DIR / "preparation_summary.json",
    rule_approvals_path: Path = GUIDANCE_APPROVALS_PATH,
    source_approvals_path: Path = GUIDANCE_SOURCE_APPROVALS_PATH,
    deployment_time: datetime | None = None,
) -> dict:
    """Stage, validate, publish, and describe one guidance release bundle."""
    if not application_version.strip():
        raise GuidanceReleaseError("application_version must be populated")
    if warning_policy not in {"report", "fail"}:
        raise GuidanceReleaseError("warning_policy must be 'report' or 'fail'")
    deployment_time = deployment_time or datetime.now(timezone.utc)
    if deployment_time.tzinfo is None or deployment_time.utcoffset() is None:
        raise GuidanceReleaseError("deployment_time must include a UTC offset")
    deployment_time = deployment_time.astimezone(timezone.utc)

    result = prepare_guidance(extracted_at=deployment_time)
    failures = result.quality.loc[result.quality.status == "FAIL"]
    warnings = result.quality.loc[result.quality.status == "WARN"]
    if not failures.empty:
        raise GuidanceReleaseError("Blocking guidance checks failed:\n" + failures.to_string(index=False))
    if warning_policy == "fail" and not warnings.empty:
        raise GuidanceReleaseError("Guidance release warnings are forbidden:\n" + warnings.to_string(index=False))

    staging_root = Path(mkdtemp(prefix="ea-guidance-release-", dir=prepared_dir.parent))
    staged_dir = staging_root / "prepared"
    try:
        paths = save_guidance_preparation(
            result,
            content_path=staged_dir / "guidance_content.parquet",
            quality_path=staged_dir / "guidance_quality.csv",
            rules_path=staged_dir / "guidance_rules.json",
            compatibility_report_path=staged_dir / "guidance_compatibility_report.csv",
            summary_path=staged_dir / "guidance_preparation_summary.json",
            candidate_review_report_path=staged_dir / "guidance_candidate_review_report.csv",
        )
        (staged_dir / ".gitkeep").touch(exist_ok=True)
        registry = load_guidance_registry(paths["rules"])
        actual_active_rule_ids = {rule.rule_id for rule in registry.active_rules}
        if actual_active_rule_ids != expected_active_rule_ids:
            raise GuidanceReleaseError(
                "Active rule IDs do not match the approved release: "
                f"expected={sorted(expected_active_rule_ids)}; actual={sorted(actual_active_rule_ids)}"
            )
        sources = sorted(
            {
                (rule.source_id, rule.source_version, rule.source_sha256)
                for rule in registry.candidate_rules
            }
        )
        active_source_ids = {rule.source_id for rule in registry.active_rules}
        model_metadata = _read_json(model_metadata_path, "structured model")
        preparation_summary = _read_json(preparation_summary_path, "structured preparation")
        artifact_hashes = {name: _sha256(path) for name, path in paths.items()}
        metadata = {
            "schema_version": "1.0",
            "application_version": application_version.strip(),
            "structured_model_version": model_metadata.get("model_version"),
            "structured_dataset_version": model_metadata.get("source_dataset_version") or preparation_summary.get("source_dataset_version"),
            "structured_dataset_sha256": model_metadata.get("dataset_sha256"),
            "guidance_registry_generated_at_utc": registry.generated_at_utc,
            "guidance_sources": [
                {
                    "source_id": source_id,
                    "source_version": version,
                    "source_sha256": digest,
                    "active": source_id in active_source_ids,
                }
                for source_id, version, digest in sources
            ],
            "expected_active_rule_ids": sorted(expected_active_rule_ids),
            "active_rule_ids": sorted(actual_active_rule_ids),
            "candidate_rule_count": len(registry.candidate_rules),
            "active_rule_count": len(registry.active_rules),
            "warning_policy": warning_policy,
            "quality_warning_count": len(warnings),
            "quality_failure_count": len(failures),
            "approver_identities": _approval_identities(
                registry.active_rules,
                rule_approvals_path,
                source_approvals_path,
            ),
            "guidance_artifact_sha256": artifact_hashes,
            "deployment_time_utc": deployment_time.isoformat(),
        }
        _atomic_json(metadata, staged_dir / GUIDANCE_RELEASE_METADATA_PATH.name)
        publish_prepared_directory(staged_dir, prepared_dir)
    except Exception:
        if staging_root.exists():
            shutil.rmtree(staging_root)
        raise
    else:
        if staging_root.exists():
            shutil.rmtree(staging_root)
    return metadata
