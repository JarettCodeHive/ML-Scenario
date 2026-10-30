import logging
import time

import pandas as pd

from cluster_c.config import INPUT_PATH
from cluster_c.preparation import prepare, save_prepared

LOGGER = logging.getLogger("prepare_data")


def main() -> None:
    started = time.perf_counter()
    LOGGER.info("Reading source data from %s", INPUT_PATH)
    raw = pd.read_csv(INPUT_PATH, low_memory=False)
    LOGGER.info("Loaded %s rows and %d columns", f"{len(raw):,}", len(raw.columns))

    LOGGER.info("Validating and preparing features")
    result = prepare(raw)
    LOGGER.info(
        "Quality checks completed: %d/%d passed",
        result.summary["quality_checks_passed"],
        result.summary["quality_checks_total"],
    )
    failed = result.checks.loc[result.checks.status == "FAIL"]
    for check in failed.itertuples():
        LOGGER.warning(
            "Non-blocking check failed: %s (%s rows) - %s",
            check.check_id,
            f"{check.failure_count:,}",
            check.description,
        )

    LOGGER.info("Saving prepared artifacts")
    paths = save_prepared(result)
    for name, path in paths.items():
        LOGGER.info("Saved %s: %s", name, path)
    LOGGER.info(
        "Preparation completed: %s rows in %.2f seconds",
        f"{len(result.frame):,}",
        time.perf_counter() - started,
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
        LOGGER.exception("Data preparation failed")
        raise SystemExit(1)
