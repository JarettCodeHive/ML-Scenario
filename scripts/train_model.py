import logging
import time

import pandas as pd

from cluster_c.config import PREPARED_FEATURE_PATH
from cluster_c.model import TRAINING_COLUMNS, save_result, sha256, train_models

FEATURE_PATH = PREPARED_FEATURE_PATH
LOGGER = logging.getLogger("train_model")


def value_counts_for_log(series: pd.Series, null_label: str = "<NULL>") -> dict[str, int]:
    """Return display-safe counts without adding values to categorical data."""
    counts: dict[str, int] = {}
    for value, count in series.value_counts(dropna=False).items():
        if not count:
            continue
        label = null_label if pd.isna(value) else str(value)
        counts[label] = counts.get(label, 0) + int(count)
    return counts


def main() -> None:
    started = time.perf_counter()
    LOGGER.info("Reading prepared features from %s", FEATURE_PATH)
    frame = pd.read_parquet(
        FEATURE_PATH,
        columns=TRAINING_COLUMNS,
        engine="pyarrow",
    )
    LOGGER.info("Loaded %s rows and %d columns", f"{len(frame):,}", len(frame.columns))
    valid_start_dates = frame["TERM_START_DATE"].dropna()
    LOGGER.info(
        "TERM_START_DATE range: %s to %s (%d null/unparseable)",
        valid_start_dates.min().date() if not valid_start_dates.empty else "N/A",
        valid_start_dates.max().date() if not valid_start_dates.empty else "N/A",
        int(frame["TERM_START_DATE"].isna().sum()),
    )
    # Parquet preserves TERM_BUCKET as a pandas categorical column. Calling
    # fillna("<NULL>") directly on it fails when <NULL> is not an existing
    # category, so count nulls without mutating or expanding the category set.
    term_counts = value_counts_for_log(frame["TERM_BUCKET"])
    LOGGER.info(
        "Term distribution (%d distinct; top 20): %s",
        len(term_counts),
        dict(list(term_counts.items())[:20]),
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
