"""Inspect prepared EA guidance without printing source text by default."""

from __future__ import annotations

import argparse
import logging

from cluster_c.unstructured.inspection import inspect_guidance_artifacts


LOGGER = logging.getLogger("inspect_ea_guidance")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-id", help="Restrict inspection to one registered source")
    parser.add_argument(
        "--show-page-text",
        action="store_true",
        help="Print extracted page text; may expose sensitive guidance content",
    )
    parser.add_argument(
        "--fail-on-warnings",
        action="store_true",
        help="Return a failure when review-required warnings remain",
    )
    args = parser.parse_args()
    result = inspect_guidance_artifacts(source_id=args.source_id)

    LOGGER.info("Preparation summary: %s", result.summary)
    LOGGER.info("Source extraction:\n%s", result.sources.to_string(index=False))
    LOGGER.info("Compatibility:\n%s", result.compatibility.to_string(index=False))
    if result.candidates.empty:
        LOGGER.warning("No candidate rules were extracted")
    else:
        columns = [
            "rule_id", "lens", "measure", "operator", "threshold",
            "comparison_measure", "scope_type", "scope_column", "scope_value",
            "source_page", "review_status",
        ]
        LOGGER.info("Candidate rules:\n%s", result.candidates[columns].to_string(index=False))
    if not result.failures.empty:
        LOGGER.error("Blocking failures:\n%s", result.failures.to_string(index=False))
    if not result.warnings.empty:
        LOGGER.warning("Review-required warnings:\n%s", result.warnings.to_string(index=False))
    if args.show_page_text:
        for page in result.pages.itertuples(index=False):
            print(f"\n--- {page.source_id} | page {page.page_number} ---\n{page.page_text}")

    if not result.failures.empty:
        raise ValueError(f"Guidance inspection found {len(result.failures)} blocking failures")
    if args.fail_on_warnings and not result.warnings.empty:
        raise ValueError(f"Guidance inspection found {len(result.warnings)} warnings")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    try:
        main()
    except Exception:
        LOGGER.exception("EA guidance inspection failed")
        raise SystemExit(1)
