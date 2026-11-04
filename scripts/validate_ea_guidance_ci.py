"""CI quality gate and determinism check for EA guidance preparation."""

from __future__ import annotations

import argparse
import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from cluster_c.unstructured.decision_rules import load_guidance_registry
from cluster_c.unstructured.preparation import prepare_guidance, save_guidance_preparation


ARTIFACT_FILENAMES = {
    "content": "guidance_content.parquet",
    "quality": "guidance_quality.csv",
    "rules": "guidance_rules.json",
    "compatibility": "guidance_compatibility_report.csv",
    "summary": "guidance_preparation_summary.json",
    "candidate_review": "guidance_candidate_review_report.csv",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_for_comparison(result, directory: Path) -> dict[str, Path]:
    return save_guidance_preparation(
        result,
        content_path=directory / ARTIFACT_FILENAMES["content"],
        quality_path=directory / ARTIFACT_FILENAMES["quality"],
        rules_path=directory / ARTIFACT_FILENAMES["rules"],
        compatibility_report_path=directory / ARTIFACT_FILENAMES["compatibility"],
        summary_path=directory / ARTIFACT_FILENAMES["summary"],
        candidate_review_report_path=directory / ARTIFACT_FILENAMES["candidate_review"],
    )


def validate_ci_guidance(
    warning_policy: str = "report",
    extracted_at: datetime | None = None,
) -> dict[str, object]:
    """Validate blocking checks, warning policy, registry, and reproducibility."""
    if warning_policy not in {"report", "fail"}:
        raise ValueError("warning_policy must be 'report' or 'fail'")
    extracted_at = extracted_at or datetime.now(timezone.utc).replace(microsecond=0)
    if extracted_at.tzinfo is None or extracted_at.utcoffset() is None:
        raise ValueError("extracted_at must include a UTC offset")

    first = prepare_guidance(extracted_at=extracted_at)
    second = prepare_guidance(extracted_at=extracted_at)
    failures = first.quality.loc[first.quality.status == "FAIL"]
    warnings = first.quality.loc[first.quality.status == "WARN"]
    if not failures.empty:
        raise ValueError("Blocking EA guidance checks failed:\n" + failures.to_string(index=False))
    if warning_policy == "fail" and not warnings.empty:
        raise ValueError("EA guidance warnings are forbidden by policy:\n" + warnings.to_string(index=False))

    with TemporaryDirectory(prefix="ea-guidance-ci-first-") as first_dir_name, TemporaryDirectory(
        prefix="ea-guidance-ci-second-"
    ) as second_dir_name:
        first_paths = _save_for_comparison(first, Path(first_dir_name))
        second_paths = _save_for_comparison(second, Path(second_dir_name))
        first_hashes = {name: _sha256(path) for name, path in first_paths.items()}
        second_hashes = {name: _sha256(path) for name, path in second_paths.items()}
        if first_hashes != second_hashes:
            mismatched = {
                name: {"first": first_hashes[name], "second": second_hashes[name]}
                for name in first_hashes
                if first_hashes[name] != second_hashes[name]
            }
            raise ValueError(f"EA guidance artifacts are not deterministic: {mismatched}")
        registry = load_guidance_registry(first_paths["rules"])

    return {
        "warning_policy": warning_policy,
        "warning_count": len(warnings),
        "failure_count": len(failures),
        "candidate_rule_count": len(registry.candidate_rules),
        "active_rule_count": len(registry.active_rules),
        "artifact_sha256": first_hashes,
        "warnings_by_check": warnings.groupby("check_id").size().sort_index().to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--warning-policy",
        choices=["report", "fail"],
        default=os.getenv("GUIDANCE_CI_WARNING_POLICY", "report"),
        help="Report warnings or make them blocking; defaults to GUIDANCE_CI_WARNING_POLICY/report",
    )
    args = parser.parse_args()
    report = validate_ci_guidance(args.warning_policy)
    print(
        "EA guidance CI validation passed: "
        f"failures={report['failure_count']}, warnings={report['warning_count']}, "
        f"candidates={report['candidate_rule_count']}, active={report['active_rule_count']}, "
        f"warning_policy={report['warning_policy']}"
    )
    for name, digest in report["artifact_sha256"].items():
        print(f"{name}: {digest}")
    if report["warnings_by_check"]:
        print("Warnings by check:")
        for check_id, count in report["warnings_by_check"].items():
            print(f"  {check_id}: {count}")


if __name__ == "__main__":
    main()
