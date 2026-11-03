"""Local CatBoost outcome model for 1-, 3-, and 5-year term scenarios."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import catboost
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from .config import MODEL_DIR, RANDOM_SEED, SOURCE_DATASET_VERSION
from .preparation import CATEGORICAL_FEATURES, EXCLUDED_IDENTIFIER_FEATURES, FEATURES, TARGET

MODEL_VERSION = "cluster-c-term-catboost-v1"
LOGGER = logging.getLogger(__name__)
TRAINING_COLUMNS = list(dict.fromkeys(
    FEATURES + [TARGET, "TERM_START_DATE", "SUBSCRIPTION_REFERENCE_ID", "CAV_END_CUSTOMER_BU_ID"]
))
TRAINING_SAMPLE_STRATA = ["FISCAL_YEAR", "TERM_BUCKET", "SCMS", "SUITE_NAME"]


def configured_thread_count() -> int:
    raw_value = os.getenv("CATBOOST_THREAD_COUNT", "2")
    try:
        thread_count = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"CATBOOST_THREAD_COUNT must be -1 or a positive integer; received {raw_value!r}"
        ) from exc
    if thread_count == 0 or thread_count < -1:
        raise ValueError(
            f"CATBOOST_THREAD_COUNT must be -1 or a positive integer; received {thread_count}"
        )
    return thread_count


def configured_max_training_rows() -> int | None:
    raw_value = os.getenv("TRAINING_SAMPLE_ROWS", "all").strip().lower()
    if raw_value in {"0", "all", "none"}:
        return None
    try:
        max_rows = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"TRAINING_SAMPLE_ROWS must be a positive integer, 'all', or 0; received {raw_value!r}"
        ) from exc
    if max_rows < 1:
        raise ValueError(
            f"TRAINING_SAMPLE_ROWS must be a positive integer, 'all', or 0; received {max_rows}"
        )
    return max_rows


@dataclass(frozen=True)
class TrainingConfig:
    iterations: int = 400
    depth: int = 6
    learning_rate: float = 0.06
    l2_leaf_reg: float = 6.0
    random_seed: int = RANDOM_SEED
    early_stopping_rounds: int = 40
    thread_count: int = field(default_factory=configured_thread_count)
    max_ctr_complexity: int = 1
    ctr_leaf_count_limit: int = 100_000
    max_training_rows: int | None = field(default_factory=configured_max_training_rows)

    def __post_init__(self) -> None:
        if self.iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if self.depth < 1:
            raise ValueError("depth must be a positive integer")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.early_stopping_rounds < 1:
            raise ValueError("early_stopping_rounds must be a positive integer")
        if self.thread_count == 0 or self.thread_count < -1:
            raise ValueError("thread_count/CATBOOST_THREAD_COUNT must be -1 or a positive integer")
        if self.max_ctr_complexity < 1:
            raise ValueError("max_ctr_complexity must be a positive integer")
        if self.ctr_leaf_count_limit < 1:
            raise ValueError("ctr_leaf_count_limit must be a positive integer")
        if self.max_training_rows is not None and self.max_training_rows < 1:
            raise ValueError("max_training_rows must be a positive integer or None")


@dataclass
class TrainingResult:
    point_model: CatBoostRegressor
    interval_model: CatBoostRegressor
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    feature_importance: pd.DataFrame
    split_summary: pd.DataFrame
    metadata: dict[str, Any]


def matrix(frame: pd.DataFrame) -> pd.DataFrame:
    # Build the model matrix directly from transformed Series. This avoids an
    # intermediate deep copy of every feature before the conversions below.
    columns = {}
    for column in CATEGORICAL_FEATURES:
        columns[column] = frame[column].astype("string").fillna("Unknown").astype(str)
    for column in set(FEATURES) - set(CATEGORICAL_FEATURES):
        columns[column] = pd.to_numeric(frame[column], errors="coerce").astype("float32")
    return pd.DataFrame(columns, index=frame.index)[FEATURES]


def chronological_split(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    LOGGER.info("Creating chronological train/validation/test split")
    # Parse only the date Series instead of copying the complete input frame.
    term_start_dates = pd.to_datetime(frame["TERM_START_DATE"], errors="coerce")
    missing_date_rows = int(term_start_dates.isna().sum())
    if missing_date_rows:
        LOGGER.warning(
            "Excluding %s rows with null or unparseable TERM_START_DATE from chronological splits",
            f"{missing_date_rows:,}",
        )
    dates = np.array(sorted(term_start_dates.dropna().unique()))
    if len(dates) < 3:
        raise ValueError("At least 3 distinct non-null term start dates are required")
    if len(dates) < 12:
        LOGGER.warning(
            "Only %d distinct term start dates are available; training will continue, "
            "but validation and test metrics are not statistically reliable",
            len(dates),
        )

    # Split on distinct dates and always reserve at least one date for both
    # validation and test, including for small diagnostic datasets.
    train_date_count = min(max(1, int(len(dates) * 0.70)), len(dates) - 2)
    validation_end_count = min(
        max(train_date_count + 1, int(len(dates) * 0.85)),
        len(dates) - 1,
    )
    train_cutoff = pd.Timestamp(dates[train_date_count - 1])
    validation_cutoff = pd.Timestamp(dates[validation_end_count - 1])
    train_mask = term_start_dates.le(train_cutoff)
    validation_mask = term_start_dates.gt(train_cutoff) & term_start_dates.le(validation_cutoff)
    test_mask = term_start_dates.gt(validation_cutoff)

    def select_split(mask: pd.Series) -> pd.DataFrame:
        part = frame.loc[mask]
        if not pd.api.types.is_datetime64_any_dtype(frame["TERM_START_DATE"]):
            part = part.assign(TERM_START_DATE=term_start_dates.loc[mask])
        return part

    splits = {
        "train": select_split(train_mask),
        "validation": select_split(validation_mask),
        "test": select_split(test_mask),
    }
    LOGGER.info(
        "Split cutoffs: train through %s; validation through %s",
        train_cutoff.date(),
        validation_cutoff.date(),
    )
    LOGGER.info("Split row counts: %s", {name: len(part) for name, part in splits.items()})
    if any(part.empty for part in splits.values()):
        raise ValueError(f"Chronological split produced an empty period: { {k: len(v) for k, v in splits.items()} }")
    for name, part in splits.items():
        missing_terms = {"1 Year", "3 Year", "5 Year"} - set(part["TERM_BUCKET"])
        if missing_terms:
            observed_terms = sorted(part["TERM_BUCKET"].dropna().astype(str).unique().tolist())
            LOGGER.warning(
                "%s split lacks term overlap for %s; observed TERM_BUCKET values: %s. "
                "Affected per-term metrics will be unavailable",
                name,
                sorted(missing_terms),
                observed_terms,
            )
    return splits


def representative_training_sample(
    frame: pd.DataFrame,
    max_rows: int | None,
    random_seed: int,
) -> pd.DataFrame:
    """Return an exact-size deterministic sample distributed across key strata."""
    if max_rows is None or len(frame) <= max_rows:
        return frame

    missing_strata = set(TRAINING_SAMPLE_STRATA) - set(frame.columns)
    if missing_strata:
        raise ValueError(f"Training sample strata are missing: {sorted(missing_strata)}")

    grouped = frame.groupby(
        TRAINING_SAMPLE_STRATA,
        dropna=False,
        observed=True,
        sort=True,
    )
    group_positions = list(grouped.indices.values())
    group_sizes = np.asarray([len(positions) for positions in group_positions], dtype=np.int64)
    group_count = len(group_sizes)

    # Preserve at least one row from every observed stratum whenever the sample
    # budget permits, then allocate the remainder proportionally.
    minimum = np.ones(group_count, dtype=np.int64) if max_rows >= group_count else np.zeros(group_count, dtype=np.int64)
    capacity = group_sizes - minimum
    remaining = max_rows - int(minimum.sum())
    if int(capacity.sum()) == 0:
        quotas = minimum
    else:
        exact_extra = capacity / capacity.sum() * remaining
        extra = np.floor(exact_extra).astype(np.int64)
        quotas = minimum + extra
        rows_left = max_rows - int(quotas.sum())
        if rows_left:
            fractional_order = np.argsort(-(exact_extra - extra))
            for position in fractional_order:
                if rows_left == 0:
                    break
                if quotas[position] < group_sizes[position]:
                    quotas[position] += 1
                    rows_left -= 1

    rng = np.random.default_rng(random_seed)
    selected = [
        rng.choice(positions, size=int(quota), replace=False)
        for positions, quota in zip(group_positions, quotas)
        if quota
    ]
    selected_positions = np.sort(np.concatenate(selected))
    sampled = frame.iloc[selected_positions]
    if len(sampled) != max_rows:
        raise RuntimeError(f"Expected {max_rows} sampled rows; received {len(sampled)}")
    return sampled


def metrics(actual, predicted) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.size == 0:
        return {"mae": np.nan, "rmse": np.nan, "wape_pct": np.nan, "bias_pct": np.nan}
    error = predicted - actual
    absolute_total = float(np.abs(actual).sum())
    signed_total = float(actual.sum())
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "wape_pct": float(np.abs(error).sum() / absolute_total * 100) if absolute_total else np.nan,
        "bias_pct": float(error.sum() / signed_total * 100) if signed_total else np.nan,
    }


def _new_model(config: TrainingConfig, loss: str) -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=config.iterations,
        depth=config.depth,
        learning_rate=config.learning_rate,
        l2_leaf_reg=config.l2_leaf_reg,
        loss_function=loss,
        random_seed=config.random_seed,
        allow_writing_files=False,
        thread_count=config.thread_count,
        max_ctr_complexity=config.max_ctr_complexity,
        ctr_leaf_count_limit=config.ctr_leaf_count_limit,
        verbose=False,
    )


def _fit(model, train_pool, validation_pool, config):
    model.fit(train_pool, eval_set=validation_pool, early_stopping_rounds=config.early_stopping_rounds, verbose=False)
    return model


def train_models(
    frame: pd.DataFrame,
    dataset_sha256: str | None = None,
    config: TrainingConfig | None = None,
) -> TrainingResult:
    if config is None:
        config = TrainingConfig()
    LOGGER.info(
        "Starting model training with %s rows, %d features, seed %d, and %d CatBoost threads",
        f"{len(frame):,}",
        len(FEATURES),
        config.random_seed,
        config.thread_count,
    )
    LOGGER.info(
        "Categorical controls: max_ctr_complexity=%d, ctr_leaf_count_limit=%s",
        config.max_ctr_complexity,
        f"{config.ctr_leaf_count_limit:,}",
    )
    LOGGER.info(
        "Boosting configuration: iterations=%d, depth=%d, learning_rate=%.3f, "
        "early_stopping_rounds=%d",
        config.iterations,
        config.depth,
        config.learning_rate,
        config.early_stopping_rounds,
    )
    splits = chronological_split(frame)
    source_train_rows = len(splits["train"])
    splits["train"] = representative_training_sample(
        splits["train"],
        max_rows=config.max_training_rows,
        random_seed=config.random_seed,
    )
    sampled_train_rows = len(splits["train"])
    if sampled_train_rows < source_train_rows:
        LOGGER.info(
            "Representative training sample: %s of %s rows (%.2f%%) across %d strata",
            f"{sampled_train_rows:,}",
            f"{source_train_rows:,}",
            sampled_train_rows / source_train_rows * 100,
            splits["train"].groupby(TRAINING_SAMPLE_STRATA, dropna=False, observed=True).ngroups,
        )
    else:
        LOGGER.info("Training sample cap not applied; using all %s training rows", f"{source_train_rows:,}")
    LOGGER.info("Building train and validation CatBoost pools")
    pools = {
        name: Pool(
            matrix(splits[name]),
            label=splits[name][TARGET].astype("float32"),
            cat_features=CATEGORICAL_FEATURES,
            feature_names=FEATURES,
        )
        for name in ("train", "validation")
    }
    model_started = time.perf_counter()
    LOGGER.info("Training point model (MAE)")
    point = _fit(_new_model(config, "MAE"), pools["train"], pools["validation"], config)
    LOGGER.info("Point model finished at iteration %d", point.get_best_iteration())

    LOGGER.info("Training interval model (MultiQuantile P10/P90)")
    interval = _fit(
        _new_model(config, "MultiQuantile:alpha=0.1,0.9"),
        pools["train"],
        pools["validation"],
        config,
    )
    LOGGER.info("Interval model finished at iteration %d", interval.get_best_iteration())
    LOGGER.info("All model fits completed in %.2f seconds", time.perf_counter() - model_started)

    train = splits["train"]
    term_suite_medians = train.groupby(["TERM_BUCKET", "SUITE_NAME"])[TARGET].median()
    global_median = float(train[TARGET].median())
    metric_rows = []
    prediction_rows = []
    for split_name in ("validation", "test"):
        LOGGER.info("Evaluating %s split", split_name)
        part = splits[split_name]
        actual = part[TARGET].to_numpy(float)
        values = matrix(part)
        central = np.clip(point.predict(values), 0, None)
        interval_raw = np.clip(interval.predict(values), 0, None)
        if interval_raw.ndim != 2 or interval_raw.shape[1] != 2:
            raise ValueError(f"Expected MultiQuantile predictions with shape (rows, 2); received {interval_raw.shape}")
        low_raw = interval_raw[:, 0]
        high_raw = interval_raw[:, 1]
        low = np.minimum.reduce([low_raw, central, high_raw])
        high = np.maximum.reduce([low_raw, central, high_raw])
        baseline = np.array([
            term_suite_medians.get((row.TERM_BUCKET, row.SUITE_NAME), global_median)
            for row in part.itertuples()
        ])
        for method, predicted in {"catboost": central, "term_suite_median": baseline}.items():
            metric_rows.append({"split": split_name, "term_bucket": "ALL", "method": method, **metrics(actual, predicted)})
        for term in ("1 Year", "3 Year", "5 Year"):
            mask = part["TERM_BUCKET"].eq(term).to_numpy()
            metric_rows.append({"split": split_name, "term_bucket": term, "method": "catboost", **metrics(actual[mask], central[mask])})
        coverage = float(np.mean((actual >= low) & (actual <= high)) * 100)
        metric_rows.append({"split": split_name, "term_bucket": "ALL", "method": "catboost_10_90_interval", "mae": np.nan, "rmse": np.nan, "wape_pct": np.nan, "bias_pct": np.nan, "interval_coverage_pct": coverage})
        out = part[["SUBSCRIPTION_REFERENCE_ID", "TERM_START_DATE", "TERM_BUCKET", TARGET]].copy()
        out["split"] = split_name
        out["prediction"] = central
        out["lower_10"] = low
        out["upper_90"] = high
        prediction_rows.append(out)

    metric_frame = pd.DataFrame(metric_rows)
    importance = pd.DataFrame({"feature": FEATURES, "importance": point.get_feature_importance(pools["train"])}).sort_values("importance", ascending=False, ignore_index=True)
    split_summary = pd.DataFrame([{
        "split": name,
        "rows": len(part),
        "customers": part["CAV_END_CUSTOMER_BU_ID"].nunique(),
        "start_date_min": part.TERM_START_DATE.min(),
        "start_date_max": part.TERM_START_DATE.max(),
        "one_year_rows": int(part.TERM_BUCKET.eq("1 Year").sum()),
        "three_year_rows": int(part.TERM_BUCKET.eq("3 Year").sum()),
        "five_year_rows": int(part.TERM_BUCKET.eq("5 Year").sum()),
    } for name, part in splits.items()])
    test_all = metric_frame[(metric_frame.split == "test") & (metric_frame.term_bucket == "ALL")]
    model_wape = float(test_all.loc[test_all.method == "catboost", "wape_pct"].iloc[0])
    baseline_wape = float(test_all.loc[test_all.method == "term_suite_median", "wape_pct"].iloc[0])
    metadata = {
        "model_version": MODEL_VERSION,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset_version": SOURCE_DATASET_VERSION,
        "dataset_sha256": dataset_sha256,
        "target": TARGET,
        "model_type": "CatBoost S-learner with MAE point and MultiQuantile P10/P90 models",
        "catboost_version": catboost.__version__,
        "features": FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "excluded_identifier_features": EXCLUDED_IDENTIFIER_FEATURES,
        "thread_count": config.thread_count,
        "training_config": {
            "iterations": config.iterations,
            "depth": config.depth,
            "learning_rate": config.learning_rate,
            "l2_leaf_reg": config.l2_leaf_reg,
            "early_stopping_rounds": config.early_stopping_rounds,
        },
        "training_sampling": {
            "enabled": sampled_train_rows < source_train_rows,
            "max_training_rows": config.max_training_rows,
            "source_training_rows": source_train_rows,
            "sampled_training_rows": sampled_train_rows,
            "sample_fraction": sampled_train_rows / source_train_rows,
            "strata": TRAINING_SAMPLE_STRATA,
            "random_seed": config.random_seed,
        },
        "max_ctr_complexity": config.max_ctr_complexity,
        "ctr_leaf_count_limit": config.ctr_leaf_count_limit,
        "best_iteration": int(point.get_best_iteration()),
        "best_iterations": {
            "point": int(point.get_best_iteration()),
            "interval": int(interval.get_best_iteration()),
        },
        "model_count": 2,
        "interval_loss": "MultiQuantile:alpha=0.1,0.9",
        "model_matrix_numeric_dtype": "float32",
        "target_training_dtype": "float32",
        "test_wape_pct": model_wape,
        "baseline_wape_pct": baseline_wape,
        "relative_wape_improvement_pct": float((baseline_wape - model_wape) / baseline_wape * 100),
        "causal_claim_supported": False,
        "local_training_only": True,
    }
    return TrainingResult(point, interval, metric_frame, pd.concat(prediction_rows, ignore_index=True), importance, split_summary, metadata)


def save_result(result: TrainingResult, output_dir: Path = MODEL_DIR) -> dict[str, Path]:
    LOGGER.info("Writing model artifacts to %s", output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "point": output_dir / "bookings_point.cbm", "interval": output_dir / "bookings_interval_10_90.cbm",
        "metadata": output_dir / "metadata.json", "metrics": output_dir / "metrics.csv", "predictions": output_dir / "holdout_predictions.csv",
        "importance": output_dir / "feature_importance.csv", "splits": output_dir / "split_summary.csv",
    }
    result.point_model.save_model(paths["point"]); result.interval_model.save_model(paths["interval"])
    paths["metadata"].write_text(json.dumps(result.metadata, indent=2) + "\n", encoding="utf-8")
    result.metrics.to_csv(paths["metrics"], index=False); result.predictions.to_csv(paths["predictions"], index=False, date_format="%Y-%m-%d")
    result.feature_importance.to_csv(paths["importance"], index=False); result.split_summary.to_csv(paths["splits"], index=False, date_format="%Y-%m-%d")
    return paths


def load_models(model_dir: Path = MODEL_DIR):
    models = []
    for filename in ("bookings_point.cbm", "bookings_interval_10_90.cbm"):
        model = CatBoostRegressor(); model.load_model(model_dir / filename); models.append(model)
    return tuple(models)


def predict_term_options(frame: pd.DataFrame, model_dir: Path = MODEL_DIR) -> pd.DataFrame:
    point, interval = load_models(model_dir)
    output = frame[["SUBSCRIPTION_REFERENCE_ID"]].copy()
    values = matrix(frame)
    for years in (1, 3, 5):
        values["TERM_BUCKET"] = f"{years} Year"
        central = np.clip(point.predict(values), 0, None)
        interval_raw = np.clip(interval.predict(values), 0, None)
        if interval_raw.ndim != 2 or interval_raw.shape[1] != 2:
            raise ValueError(f"Expected MultiQuantile predictions with shape (rows, 2); received {interval_raw.shape}")
        low_raw = interval_raw[:, 0]
        high_raw = interval_raw[:, 1]
        output[f"bookings_{years}y"] = central
        output[f"bookings_{years}y_p10"] = np.minimum.reduce([low_raw, central, high_raw])
        output[f"bookings_{years}y_p90"] = np.maximum.reduce([low_raw, central, high_raw])
    return output


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
