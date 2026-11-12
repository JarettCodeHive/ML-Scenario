"""Train and compare different models on the Cluster C bookings target.

Usage (from the project root, after setting PYTHONPATH to src):

    python scripts/compare_models.py
        Uses the prepared sample (data/prepared/term_model_features.parquet,
        built by scripts/prepare_data.py) and saves results to
        data/models/model_comparison/.

    python scripts/compare_models.py --rows 100000
        Generates a NEW synthetic dataset of that many deals in memory (same
        generator and seed as the sample), prepares it, compares the models and
        saves results to data/models/model_comparison_100000_rows/.
        data/input/term_deals.csv and the prepared sample are NOT changed.

Options:
    --rows N         generate N synthetic deals instead of using the prepared sample
    --folds K        backtest folds for the stability check (default 3, 0 = skip, faster)
    --seed S         random seed for the generated data (default 42)
    --save-data PATH also write the generated raw CSV to PATH (optional, can be large)

Every result set appears in the "Result set" picker on the Streamlit
"Model Comparison" page.
"""

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

from cluster_c.config import DATA_DIR, PREPARED_FEATURE_PATH, RANDOM_SEED
from cluster_c.model_comparison import COMPARISON_DIR, different_models, save_comparison

FEATURE_PATH = PREPARED_FEATURE_PATH
LOGGER = logging.getLogger("compare_models")


def output_dir_for(rows: int | None) -> Path:
    """Sample results keep the original folder; generated sizes get their own folder."""
    return COMPARISON_DIR if rows is None else DATA_DIR / "models" / f"model_comparison_{rows}_rows"


def load_frame(rows: int | None, seed: int, save_data: Path | None) -> pd.DataFrame:
    if rows is None:
        # 1a. The prepared 5,000-row sample (output of scripts/prepare_data.py).
        frame = pd.read_parquet(FEATURE_PATH, engine="pyarrow")
        LOGGER.info("Loaded %s rows from %s", f"{len(frame):,}", FEATURE_PATH)
        return frame
    # 1b. Generate a larger synthetic dataset in memory and prepare it the same way.
    from cluster_c.preparation import prepare
    from cluster_c.synthetic import generate_term_deals, write_term_deals

    started = time.perf_counter()
    LOGGER.info("Generating %s synthetic deals (seed %d)", f"{rows:,}", seed)
    raw = generate_term_deals(seed=seed, row_count=rows)
    if save_data is not None:
        write_term_deals(raw, save_data)
        LOGGER.info("Saved generated raw data to %s", save_data)
    frame = prepare(raw).frame
    LOGGER.info("Generated and prepared %s rows in %.1f seconds", f"{len(frame):,}", time.perf_counter() - started)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare models on the Cluster C bookings target.")
    parser.add_argument("--rows", type=int, default=None, help="generate this many synthetic deals instead of using the prepared sample")
    parser.add_argument("--folds", type=int, default=3, help="backtest folds for the stability check (0 = skip)")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="random seed for generated data")
    parser.add_argument("--save-data", type=Path, default=None, help="optional path to save the generated raw CSV")
    args = parser.parse_args()
    if args.rows is not None and args.rows < 100:
        parser.error("--rows must be at least 100")
    if args.save_data is not None and args.rows is None:
        parser.error("--save-data only applies together with --rows")
    if args.save_data is not None and args.save_data.resolve() == (DATA_DIR / "input" / "term_deals.csv").resolve():
        parser.error("--save-data must not overwrite data/input/term_deals.csv")

    started = time.perf_counter()
    frame = load_frame(args.rows, args.seed, args.save_data)
    # 2. Train and evaluate every model (see model_comparison.py).
    result = different_models(frame, n_backtest_folds=args.folds)
    result.metadata["source_rows"] = len(frame)
    result.metadata["generated_rows"] = args.rows
    result.metadata["result_set"] = "prepared sample" if args.rows is None else f"generated {args.rows:,} rows (seed {args.seed})"
    # 3. Save the result tables so the Streamlit "Model Comparison" page can show them.
    paths = save_comparison(result, output_dir_for(args.rows))
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
