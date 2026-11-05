"""Train and compare different models on the prepared Cluster C features.

Usage (from the project root):
    PYTHONPATH=src python scripts/compare_models.py
Results are written to data/models/model_comparison/ and shown on the
"Model Comparison" page of the Streamlit app.
"""

import logging
import time

import pandas as pd

from cluster_c.config import PREPARED_DIR
from cluster_c.model_comparison import different_models, save_comparison

FEATURE_PATH = PREPARED_DIR / "term_model_features.csv"
LOGGER = logging.getLogger("compare_models")


def main() -> None:
    started = time.perf_counter()
    frame = pd.read_csv(FEATURE_PATH, parse_dates=["TERM_START_DATE", "TERM_END_DATE"], low_memory=False)
    LOGGER.info("Loaded %s rows from %s", f"{len(frame):,}", FEATURE_PATH)
    result = different_models(frame)
    paths = save_comparison(result)
    columns = ["rank", "model", "test_wape_pct", "test_mae", "test_r2", "test_within_20pct",
               "backtest_wape_mean", "backtest_wape_std", "monotonic_pct", "scenario_safe", "fit_seconds"]
    LOGGER.info("Leaderboard (sorted by test WAPE, lower is better):\n%s",
                result.leaderboard[columns].round(2).to_string(index=False))
    LOGGER.info("Recommended: %s. %s", result.metadata["recommended_model"], result.metadata["recommendation_reason"])
    for name, path in paths.items():
        LOGGER.info("Saved %s: %s", name, path)
    LOGGER.info("Comparison completed in %.1f seconds", time.perf_counter() - started)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    try:
        main()
    except Exception:
        LOGGER.exception("Model comparison failed")
        raise SystemExit(1)
