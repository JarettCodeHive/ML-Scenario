"""Train and compare different models on the prepared Cluster C features.

Usage (from the project root):
    PYTHONPATH=src python scripts/compare_models.py
Results are written to data/models/model_comparison/ and shown on the
"Model Comparison" page of the Streamlit app.
"""

import logging
import time

import pandas as pd

from cluster_c.config import PREPARED_FEATURE_PATH
from cluster_c.model_comparison import different_models, save_comparison

FEATURE_PATH = PREPARED_FEATURE_PATH
LOGGER = logging.getLogger("compare_models")


def main() -> None:
    started = time.perf_counter()
    # 1. Load the prepared features (the output of scripts/prepare_data.py).
    frame = pd.read_parquet(FEATURE_PATH, engine="pyarrow")
    LOGGER.info("Loaded %s rows from %s", f"{len(frame):,}", FEATURE_PATH)
    # 2. Train and evaluate every model (about 1-2 minutes; see model_comparison.py).
    result = different_models(frame)
    # 3. Save the result tables so the Streamlit "Model Comparison" page can show them.
    paths = save_comparison(result)
    # 4. Print a short leaderboard and the recommendation to the terminal.
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
