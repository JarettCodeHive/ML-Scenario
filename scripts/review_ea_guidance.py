"""Initialize and validate the controlled Phase 5 candidate-review records."""

from __future__ import annotations

import argparse
import logging

from cluster_c.config import GUIDANCE_RULES_PATH
from cluster_c.unstructured.candidate_review import (
    initialize_candidate_reviews,
    save_candidate_review_report,
    validate_candidate_reviews,
)
from cluster_c.unstructured.decision_rules import load_guidance_registry


LOGGER = logging.getLogger("review_ea_guidance")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--initialize",
        action="store_true",
        help="Append missing in-review records without changing existing human decisions",
    )
    parser.add_argument("--source-id", help="Restrict the displayed report to one source")
    parser.add_argument(
        "--fail-on-incomplete",
        action="store_true",
        help="Return a failure until every candidate and the precedence policy are reviewed",
    )
    args = parser.parse_args()

    registry = load_guidance_registry(GUIDANCE_RULES_PATH)
    if args.initialize:
        added = initialize_candidate_reviews(registry.candidate_rules)
        LOGGER.info("Added %d missing candidate-review records", added)

    result = validate_candidate_reviews(registry.candidate_rules)
    report_path = save_candidate_review_report(result)
    display = result.report
    if args.source_id:
        known = set(display["source_id"]) if not display.empty else set()
        if args.source_id not in known:
            raise ValueError(f"Unknown guidance source_id: {args.source_id}")
        display = display.loc[display.source_id == args.source_id]

    columns = [
        "rule_id",
        "lens",
        "measure",
        "operator",
        "threshold",
        "comparison_measure",
        "scope_type",
        "scope_column",
        "scope_value",
        "source_page",
        "review_decision",
        "review_complete",
        "activation_review_eligible",
    ]
    LOGGER.info("Candidate review report saved to %s", report_path)
    LOGGER.info("Lens-status precedence confirmed: %s", result.precedence_confirmed)
    if display.empty:
        LOGGER.warning("No candidate rules are available for review")
    else:
        LOGGER.info("Candidate review status:\n%s", display[columns].to_string(index=False))

    failures = result.checks.loc[result.checks.status == "FAIL"]
    warnings = result.checks.loc[result.checks.status == "WARN"]
    if not failures.empty:
        LOGGER.error("Candidate-review failures:\n%s", failures.to_string(index=False))
        raise ValueError(f"Candidate review validation found {len(failures)} blocking failures")
    if not warnings.empty:
        LOGGER.warning("Incomplete candidate reviews:\n%s", warnings.to_string(index=False))
    if args.fail_on_incomplete and not result.is_complete:
        raise ValueError("Phase 5 candidate review is not complete")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    try:
        main()
    except Exception:
        LOGGER.exception("EA guidance candidate review failed")
        raise SystemExit(1)
