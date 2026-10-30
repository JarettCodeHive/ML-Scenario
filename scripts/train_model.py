import logging
import time

import pandas as pd

from cluster_c.config import PREPARED_DIR
from cluster_c.model import save_result, sha256, train_models

FEATURE_PATH = PREPARED_DIR / "term_model_features.csv"
LOGGER = logging.getLogger("train_model")


def main() -> None:
    started = time.perf_counter()
    LOGGER.info("Reading prepared features from %s", FEATURE_PATH)
    frame = pd.read_csv(
        FEATURE_PATH,
        parse_dates=["TERM_START_DATE", "TERM_END_DATE"],
        low_memory=False,
    )
    LOGGER.info("Loaded %s rows and %d columns", f"{len(frame):,}", len(frame.columns))
    valid_start_dates = frame["TERM_START_DATE"].dropna()
    LOGGER.info(
        "TERM_START_DATE range: %s to %s (%d null/unparseable)",
        valid_start_dates.min().date() if not valid_start_dates.empty else "N/A",
        valid_start_dates.max().date() if not valid_start_dates.empty else "N/A",
        int(frame["TERM_START_DATE"].isna().sum()),
    )
    term_counts = frame["TERM_BUCKET"].fillna("<NULL>").value_counts()
    LOGGER.info(
        "Term distribution (%d distinct; top 20): %s",
        len(term_counts),
        term_counts.head(20).to_dict(),
    )

    dataset_hash = sha256(FEATURE_PATH)
    LOGGER.info("Dataset SHA-256: %s", dataset_hash)
    result = train_models(frame, dataset_sha256=dataset_hash)

    LOGGER.info("Saving trained models and evaluation artifacts")
    paths = save_result(result)
    LOGGER.info("Split summary:\n%s", result.split_summary.to_string(index=False))
    LOGGER.info("Metrics:\n%s", result.metrics.to_string(index=False))
    for name, path in paths.items():
        LOGGER.info("Saved %s: %s", name, path)
    LOGGER.info("Training completed in %.2f seconds", time.perf_counter() - started)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    try:
        main()
    except Exception:
        LOGGER.exception("Model training failed")
        raise SystemExit(1)
