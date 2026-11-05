"""Copy and register a new real EA guidance PDF as an in-review source."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from cluster_c.unstructured.compatibility import ALLOWED_ONBOARDING_METHODS
from cluster_c.unstructured.registration import register_guidance_source


LOGGER = logging.getLogger("register_ea_guidance")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", required=True, type=Path, help="Source PDF to copy into the repository")
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--effective-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--business-owner", required=True)
    parser.add_argument("--technical-owner", required=True)
    parser.add_argument("--authorized-rule-reviewer", required=True)
    parser.add_argument("--backup-reviewer", required=True)
    parser.add_argument("--source-repository", required=True)
    parser.add_argument("--governance-contact", required=True)
    parser.add_argument("--retention-policy", required=True)
    parser.add_argument("--access-requirements", required=True)
    parser.add_argument("--compatibility-assessor", required=True)
    parser.add_argument("--onboarding-method", required=True, choices=sorted(ALLOWED_ONBOARDING_METHODS))
    parser.add_argument("--assessment-notes", default="")
    parser.add_argument("--review-frequency-days", type=int, default=365)
    parser.add_argument("--supersedes-source-id", default="")
    parser.add_argument(
        "--confirm-permitted-use",
        action="store_true",
        help="Confirm permission for storage, text extraction, rule derivation, and Streamlit source reference",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = register_guidance_source(
        args.pdf,
        source_id=args.source_id,
        title=args.title,
        version=args.version,
        effective_date=args.effective_date,
        business_owner=args.business_owner,
        technical_owner=args.technical_owner,
        authorized_rule_reviewer=args.authorized_rule_reviewer,
        backup_reviewer=args.backup_reviewer,
        source_repository=args.source_repository,
        governance_contact=args.governance_contact,
        retention_policy=args.retention_policy,
        access_requirements=args.access_requirements,
        compatibility_assessor=args.compatibility_assessor,
        selected_onboarding_method=args.onboarding_method,
        assessment_notes=args.assessment_notes,
        permitted_use_confirmed=args.confirm_permitted_use,
        review_frequency_days=args.review_frequency_days,
        supersedes_source_id=args.supersedes_source_id,
        dry_run=args.dry_run,
    )
    action = "Would register" if result.dry_run else "Registered"
    LOGGER.info("%s %s at %s", action, result.source_id, result.destination)
    LOGGER.info("SHA-256: %s", result.source_sha256)
    LOGGER.info("Initial review status: in_review")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    try:
        main()
    except Exception:
        LOGGER.exception("EA guidance registration failed")
        raise SystemExit(1)
