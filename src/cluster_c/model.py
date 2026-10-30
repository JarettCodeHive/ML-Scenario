"""Local CatBoost outcome model for 1-, 3-, and 5-year term scenarios."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import catboost
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from .config import MODEL_DIR, RANDOM_SEED, SOURCE_DATASET_VERSION
from .preparation import CATEGORICAL_FEATURES, FEATURES, TARGET

MODEL_VERSION = "cluster-c-term-catboost-v1"


@dataclass(frozen=True)
class TrainingConfig:
    iterations: int = 650
    depth: int = 7
    learning_rate: float = 0.04
    l2_leaf_reg: float = 6.0
    random_seed: int = RANDOM_SEED
    early_stopping_rounds: int = 70


@dataclass
class TrainingResult:
    point_model: CatBoostRegressor
    lower_model: CatBoostRegressor
    upper_model: CatBoostRegressor
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    feature_importance: pd.DataFrame
    split_summary: pd.DataFrame
    metadata: dict[str, Any]


def matrix(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame[FEATURES].copy()
    for column in CATEGORICAL_FEATURES:
        output[column] = output[column].fillna("Unknown").astype(str)
    for column in set(FEATURES) - set(CATEGORICAL_FEATURES):
        output[column] = pd.to_numeric(output[column], errors="coerce")
    return output


def chronological_split(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    data = frame.copy()
    data["TERM_START_DATE"] = pd.to_datetime(data["TERM_START_DATE"])
    dates = np.array(sorted(data["TERM_START_DATE"].dropna().unique()))
    if len(dates) < 12:
        raise ValueError("At least 12 distinct term start dates are required")
    train_cutoff = pd.Timestamp(dates[int(len(dates) * 0.70) - 1])
    validation_cutoff = pd.Timestamp(dates[int(len(dates) * 0.85) - 1])
    splits = {
        "train": data[data.TERM_START_DATE <= train_cutoff].copy(),
        "validation": data[(data.TERM_START_DATE > train_cutoff) & (data.TERM_START_DATE <= validation_cutoff)].copy(),
        "test": data[data.TERM_START_DATE > validation_cutoff].copy(),
    }
    if any(part.empty for part in splits.values()):
        raise ValueError(f"Chronological split produced an empty period: { {k: len(v) for k, v in splits.items()} }")
    for name, part in splits.items():
        missing_terms = {"1 Year", "3 Year", "5 Year"} - set(part["TERM_BUCKET"])
        if missing_terms:
            raise ValueError(f"{name} split lacks term overlap for {sorted(missing_terms)}")
    return splits


def metrics(actual, predicted) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - actual
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "wape_pct": float(np.abs(error).sum() / np.abs(actual).sum() * 100),
        "bias_pct": float(error.sum() / actual.sum() * 100),
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
        thread_count=1,
        verbose=False,
    )


def _fit(model, train_pool, validation_pool, config):
    model.fit(train_pool, eval_set=validation_pool, early_stopping_rounds=config.early_stopping_rounds, verbose=False)
    return model


def train_models(frame: pd.DataFrame, dataset_sha256: str | None = None, config: TrainingConfig = TrainingConfig()) -> TrainingResult:
    splits = chronological_split(frame)
    pools = {
        name: Pool(matrix(part), label=part[TARGET].astype(float), cat_features=CATEGORICAL_FEATURES, feature_names=FEATURES)
        for name, part in splits.items()
    }
    point = _fit(_new_model(config, "MAE"), pools["train"], pools["validation"], config)
    lower = _fit(_new_model(config, "Quantile:alpha=0.1"), pools["train"], pools["validation"], config)
    upper = _fit(_new_model(config, "Quantile:alpha=0.9"), pools["train"], pools["validation"], config)

    train = splits["train"]
    term_suite_medians = train.groupby(["TERM_BUCKET", "SUITE_NAME"])[TARGET].median()
    global_median = float(train[TARGET].median())
    metric_rows = []
    prediction_rows = []
    for split_name in ("validation", "test"):
        part = splits[split_name]
        actual = part[TARGET].to_numpy(float)
        central = np.clip(point.predict(matrix(part)), 0, None)
        low_raw = np.clip(lower.predict(matrix(part)), 0, None)
        high_raw = np.clip(upper.predict(matrix(part)), 0, None)
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
        "model_type": "CatBoost S-learner with TERM_BUCKET as treatment feature",
        "catboost_version": catboost.__version__,
        "features": FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "best_iteration": int(point.get_best_iteration()),
        "test_wape_pct": model_wape,
        "baseline_wape_pct": baseline_wape,
        "relative_wape_improvement_pct": float((baseline_wape - model_wape) / baseline_wape * 100),
        "causal_claim_supported": False,
        "local_training_only": True,
    }
    return TrainingResult(point, lower, upper, metric_frame, pd.concat(prediction_rows, ignore_index=True), importance, split_summary, metadata)


def save_result(result: TrainingResult, output_dir: Path = MODEL_DIR) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "point": output_dir / "bookings_point.cbm", "lower": output_dir / "bookings_lower_10.cbm", "upper": output_dir / "bookings_upper_90.cbm",
        "metadata": output_dir / "metadata.json", "metrics": output_dir / "metrics.csv", "predictions": output_dir / "holdout_predictions.csv",
        "importance": output_dir / "feature_importance.csv", "splits": output_dir / "split_summary.csv",
    }
    result.point_model.save_model(paths["point"]); result.lower_model.save_model(paths["lower"]); result.upper_model.save_model(paths["upper"])
    paths["metadata"].write_text(json.dumps(result.metadata, indent=2) + "\n", encoding="utf-8")
    result.metrics.to_csv(paths["metrics"], index=False); result.predictions.to_csv(paths["predictions"], index=False, date_format="%Y-%m-%d")
    result.feature_importance.to_csv(paths["importance"], index=False); result.split_summary.to_csv(paths["splits"], index=False, date_format="%Y-%m-%d")
    return paths


def load_models(model_dir: Path = MODEL_DIR):
    models = []
    for filename in ("bookings_point.cbm", "bookings_lower_10.cbm", "bookings_upper_90.cbm"):
        model = CatBoostRegressor(); model.load_model(model_dir / filename); models.append(model)
    return tuple(models)


def predict_term_options(frame: pd.DataFrame, model_dir: Path = MODEL_DIR) -> pd.DataFrame:
    point, lower, upper = load_models(model_dir)
    output = frame[["SUBSCRIPTION_REFERENCE_ID"]].copy()
    for years in (1, 3, 5):
        scenario = frame.copy(); scenario["TERM_BUCKET"] = f"{years} Year"
        values = matrix(scenario)
        central = np.clip(point.predict(values), 0, None); low_raw = np.clip(lower.predict(values), 0, None); high_raw = np.clip(upper.predict(values), 0, None)
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
