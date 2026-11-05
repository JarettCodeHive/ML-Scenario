"""Safely record hash-bound Phase 6 rule and source approval decisions."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime

from cluster_c.unstructured.approval_recording import (
    record_rule_decision,
    record_source_decision,
)


LOGGER = logging.getLogger("record_ea_guidance_approval")


def _timestamp(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Timestamp must be ISO-8601 with a UTC offset") from exc


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--decision", required=True, choices=["approved", "rejected"])
    parser.add_argument("--reviewed-by", required=True)
    parser.add_argument("--review-notes", required=True)
    parser.add_argument("--reviewed-at", type=_timestamp, help="ISO-8601 timestamp; defaults to current UTC")
    parser.add_argument("--dry-run", action="store_true", help="Validate without updating registries")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="subject", required=True)
    rule_parser = commands.add_parser("rule", help="Record an individual rule decision")
    rule_parser.add_argument("--rule-id", required=True)
    _common_arguments(rule_parser)
    source_parser = commands.add_parser("source", help="Record a document-level source decision")
    source_parser.add_argument("--source-id", required=True)
    _common_arguments(source_parser)
    args = parser.parse_args()

    common = {
        "decision": args.decision,
        "reviewed_by": args.reviewed_by,
        "review_notes": args.review_notes,
        "reviewed_at": args.reviewed_at,
        "dry_run": args.dry_run,
    }
    if args.subject == "rule":
        result = record_rule_decision(args.rule_id, **common)
    else:
        result = record_source_decision(args.source_id, **common)
    LOGGER.info(
        "%s decision validated%s: subject=%s decision=%s reviewer=%s timestamp=%s",
        args.subject.title(),
        " (dry run)" if result.dry_run else " and recorded",
        result.subject_id,
        result.decision,
        result.reviewer,
        result.reviewed_at_utc,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    try:
        main()
    except Exception:
        LOGGER.exception("EA guidance approval recording failed")
        raise SystemExit(1)
