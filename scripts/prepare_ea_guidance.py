"""Validate and extract page-level content from general EA guidance PDFs."""

from __future__ import annotations

import logging
import time

from cluster_c.config import GUIDANCE_MANIFEST_PATH
from cluster_c.unstructured.discovery import refresh_guidance_manifest
from cluster_c.unstructured.preparation import prepare_guidance, save_guidance_preparation


LOGGER = logging.getLogger("prepare_ea_guidance")


def main() -> None:
    started = time.perf_counter()
    manifest = refresh_guidance_manifest()
    LOGGER.info("Discovered %d PDF source(s) and refreshed %s", len(manifest), GUIDANCE_MANIFEST_PATH)
    LOGGER.info("Reading EA guidance manifest from %s", GUIDANCE_MANIFEST_PATH)
    result = prepare_guidance()
    paths = save_guidance_preparation(result)
    for name, path in paths.items():
        LOGGER.info("Saved %s: %s", name, path)
    LOGGER.info(
        "Guidance preparation completed: %d sources, %d pages, %d candidates, "
        "%d active rules, %d warnings, %d failed checks in %.2f seconds",
        result.summary["extracted_source_count"],
        result.summary["extracted_page_count"],
        result.summary["candidate_rule_count"],
        result.summary["active_rule_count"],
        result.summary["quality_warning_count"],
        result.summary["quality_failure_count"],
        time.perf_counter() - started,
    )
    if not result.is_valid:
        failed = result.quality.loc[result.quality.status == "FAIL", ["check_id", "source_id", "detail"]]
        raise ValueError("EA guidance preparation failed:\n" + failed.to_string(index=False))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    try:
        main()
    except Exception:
        LOGGER.exception("EA guidance preparation failed")
        raise SystemExit(1)
