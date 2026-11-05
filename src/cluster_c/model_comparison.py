"""Compare different regression models for the Cluster C bookings target.

``different_models()`` trains several model families on exactly the same
chronological train / validation / test split that the production CatBoost
model uses, then scores them on three things:

1. Accuracy on the untouched test period (MAE, RMSE, WAPE, bias, R², MAPE,
   share of deals predicted within ±20%), overall and per term bucket.
2. Stability: an expanding-window backtest over the train + validation period
   (the test period is never touched), reported as mean and spread of WAPE.
3. Scenario fitness: every test deal is re-scored as 1, 3 and 5 years, the
   same way the what-if engine does. A model that is accurate but does not
   respond sensibly to TERM_BUCKET is not safe for the Cluster C scenario.

This module is evaluation only. It does not replace the CatBoost models that
``app.py`` loads from ``data/models/cluster_c_term_v1``.
"""

from __future__ import annotations

import json
import logging
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .config import DATA_DIR, RANDOM_SEED, SOURCE_DATASET_VERSION
from .model import TrainingConfig, chronological_split, matrix
from .preparation import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET

LOGGER = logging.getLogger(__name__)

COMPARISON_DIR = DATA_DIR / "models" / "model_comparison"
TERMS = ("1 Year", "3 Year", "5 Year")
TERM_YEARS = {"1 Year": 1, "3 Year": 3, "5 Year": 5}
CURRENT_MODEL = "CatBoost (current)"
BASELINE_MODEL = "Term × suite median (baseline)"
# Identifiers with hundreds of levels: fine for CatBoost's ordered target
# statistics, but they make a one-hot linear model overfit, so the linear
# model leaves them out.
HIGH_CARDINALITY = {"CAV_END_CUSTOMER_BU_ID", "UCD_C4_ACCOUNT_ID"}
MONEY_FEATURES = ["CUSTOMER_PRIOR_AVG_ANNUALIZED_PRICE", "CUSTOMER_PRIOR_AVG_BOOKINGS"]
# The synthetic generator (synthetic.py) multiplies the annual value by these
# factors for 1/3/5-year terms, so on synthetic data the TRUE effect of the
# default scenario (100% flexible, 50% five-year) is known exactly:
# (0.5*1*0.94 + 0.5*5*1.075 - 3*1.0) / (3*1.0) = +5.25%.
# Real Cisco data has no known truth; this check is only shown for synthetic runs.
SYNTHETIC_TERM_ANNUAL_FACTOR = {1: 0.94, 3: 1.00, 5: 1.075}
SYNTHETIC_TRUE_DEFAULT_CHANGE_PCT = (
    (0.5 * 1 * SYNTHETIC_TERM_ANNUAL_FACTOR[1] + 0.5 * 5 * SYNTHETIC_TERM_ANNUAL_FACTOR[5] - 3 * SYNTHETIC_TERM_ANNUAL_FACTOR[3])
    / (3 * SYNTHETIC_TERM_ANNUAL_FACTOR[3]) * 100
)


SYNTHETIC_TRUTH_LABEL = "Known truth (synthetic data)"


def is_synthetic_dataset(version: str = SOURCE_DATASET_VERSION) -> bool:
    return "synthetic" in str(version).lower()


# A model is flagged "scenario-unsafe" when fewer test deals than this keep
# the expected order bookings(1y) < bookings(3y) < bookings(5y).
MONOTONIC_THRESHOLD_PCT = 95.0


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def regression_metrics(actual, predicted) -> dict[str, float]:
    """Accuracy metrics used on the leaderboard (WAPE is the primary one)."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    keys = ("mae", "rmse", "wape_pct", "bias_pct", "r2", "mape_pct", "within_20pct")
    if actual.size == 0:
        return {key: np.nan for key in keys}
    error = predicted - actual
    abs_total = float(np.abs(actual).sum())
    total = float(actual.sum())
    ss_tot = float(np.square(actual - actual.mean()).sum())
    nonzero = actual != 0
    ape = np.abs(error[nonzero] / actual[nonzero])
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "wape_pct": float(np.abs(error).sum() / abs_total * 100) if abs_total else np.nan,
        "bias_pct": float(error.sum() / total * 100) if total else np.nan,
        "r2": float(1 - np.square(error).sum() / ss_tot) if ss_tot else np.nan,
        "mape_pct": float(ape.mean() * 100) if ape.size else np.nan,
        "within_20pct": float((ape <= 0.20).mean() * 100) if ape.size else np.nan,
    }


# --------------------------------------------------------------------------
# Feature encodings for the non-CatBoost libraries
# --------------------------------------------------------------------------
class _OrdinalEncoder:
    """Category -> integer code learned on the training rows; unseen -> -1."""

    def __init__(self) -> None:
        self.maps: dict[str, dict[str, int]] = {}

    def fit(self, frame: pd.DataFrame) -> "_OrdinalEncoder":
        base = matrix(frame)
        for column in CATEGORICAL_FEATURES:
            levels = sorted(base[column].unique())
            self.maps[column] = {level: code for code, level in enumerate(levels)}
        return self

    def transform(self, frame: pd.DataFrame, as_category: bool = False) -> pd.DataFrame:
        out = matrix(frame)
        for column in CATEGORICAL_FEATURES:
            codes = out[column].map(self.maps[column]).fillna(-1).astype(int)
            if as_category and column not in HIGH_CARDINALITY:
                # Fixed category list so train/validation/test share one coding.
                # Customer/account IDs stay integer codes: native categorical
                # splits on ~700 levels overfit badly (XGBoost test WAPE 22% -> 15%).
                n_levels = len(self.maps[column])
                out[column] = pd.Categorical(codes.where(codes >= 0), categories=range(n_levels))
            else:
                out[column] = codes
        # Ordinal codes sort "1 Year" < "3 Year" < "5 Year", but the numeric
        # term length is clearer for every model, so add it explicitly.
        out["TERM_YEARS_NUM"] = matrix(frame)["TERM_BUCKET"].map(TERM_YEARS).astype(float)
        return out


class _OneHotEncoder:
    """Scaled numeric + one-hot categoricals for the linear model."""

    def __init__(self) -> None:
        self.categories: dict[str, list[str]] = {}
        self.columns: list[str] = []
        self.medians: dict[str, float] = {}
        self.mean: pd.Series | None = None
        self.std: pd.Series | None = None

    def _numeric(self, frame: pd.DataFrame) -> pd.DataFrame:
        base = matrix(frame)
        numeric = base[[c for c in FEATURES if c in NUMERIC_FEATURES]].astype(float).copy()
        for column in MONEY_FEATURES:
            numeric[column + "_MISSING"] = numeric[column].isna().astype(float)
            numeric[column] = np.log1p(numeric[column].clip(lower=0))
        return numeric

    def fit(self, frame: pd.DataFrame) -> "_OneHotEncoder":
        base = matrix(frame)
        for column in CATEGORICAL_FEATURES:
            if column not in HIGH_CARDINALITY:
                self.categories[column] = sorted(base[column].unique())
        numeric = self._numeric(frame)
        self.medians = numeric.median().fillna(0).to_dict()
        encoded = self._encode(frame, fitting=True)
        self.columns = list(encoded.columns)
        return self

    def _encode(self, frame: pd.DataFrame, fitting: bool = False) -> pd.DataFrame:
        base = matrix(frame)
        numeric = self._numeric(frame).fillna(self.medians)
        dummies = [
            pd.get_dummies(pd.Categorical(base[col], categories=levels), prefix=col, dtype=float).set_index(base.index)
            for col, levels in self.categories.items()
        ]
        encoded = pd.concat([numeric, *dummies], axis=1)
        if fitting:
            self.mean = encoded.mean()
            self.std = encoded.std().replace(0, 1).fillna(1)
        return (encoded - self.mean) / self.std

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self._encode(frame).reindex(columns=self.columns, fill_value=0.0)


# --------------------------------------------------------------------------
# Model wrappers: fit(train, validation | None) -> self, predict(frame) -> array
# --------------------------------------------------------------------------
@dataclass
class ModelSpec:
    name: str
    family: str
    description: str
    builder: Callable[[], Any]
    uses_validation: bool = False


class _Baseline:
    """Median bookings by term and suite: the 'is ML worth it?' reference."""

    def fit(self, train, validation=None):
        self.medians = train.groupby(["TERM_BUCKET", "SUITE_NAME"])[TARGET].median()
        self.global_median = float(train[TARGET].median())
        return self

    def predict(self, frame):
        return np.array([
            self.medians.get((row.TERM_BUCKET, row.SUITE_NAME), self.global_median)
            for row in frame[["TERM_BUCKET", "SUITE_NAME"]].itertuples()
        ], dtype=float)


class _CatBoost:
    """Same CatBoost settings as model.py; optionally learns the annual value."""

    def __init__(self, annualized_target: bool = False, config: TrainingConfig = TrainingConfig()):
        self.annualized_target = annualized_target
        self.config = config
        self.best_iteration: int | None = None

    def _label(self, frame):
        label = frame[TARGET].astype(float)
        return label / frame["TERM_BUCKET"].map(TERM_YEARS) if self.annualized_target else label

    def fit(self, train, validation=None):
        from catboost import CatBoostRegressor, Pool

        cfg = self.config
        iterations = cfg.iterations if validation is not None or self.best_iteration is None else self.best_iteration + 1
        self.model = CatBoostRegressor(
            iterations=iterations, depth=cfg.depth, learning_rate=cfg.learning_rate,
            l2_leaf_reg=cfg.l2_leaf_reg, loss_function="MAE", random_seed=cfg.random_seed,
            allow_writing_files=False, thread_count=1, verbose=False,
        )
        pool = Pool(matrix(train), label=self._label(train), cat_features=CATEGORICAL_FEATURES)
        if validation is not None:
            eval_pool = Pool(matrix(validation), label=self._label(validation), cat_features=CATEGORICAL_FEATURES)
            self.model.fit(pool, eval_set=eval_pool, early_stopping_rounds=cfg.early_stopping_rounds, verbose=False)
            self.best_iteration = int(self.model.get_best_iteration())
        else:
            self.model.fit(pool, verbose=False)
        return self

    def predict(self, frame):
        prediction = self.model.predict(matrix(frame))
        if self.annualized_target:
            prediction = prediction * frame["TERM_BUCKET"].map(TERM_YEARS).to_numpy(float)
        return prediction


class _Sklearn:
    """Ridge / Random Forest / Extra Trees / HistGradientBoosting."""

    def __init__(self, kind: str):
        self.kind = kind

    def fit(self, train, validation=None):
        from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
        from sklearn.linear_model import RidgeCV

        y = train[TARGET].astype(float).to_numpy()
        if self.kind == "ridge":
            self.encoder = _OneHotEncoder().fit(train)
            # Bookings are heavily right-skewed; a linear model fits log(bookings).
            self.model = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(self.encoder.transform(train), np.log1p(y))
            return self
        self.encoder = _OrdinalEncoder().fit(train)
        x = self.encoder.transform(train)
        if self.kind == "random_forest":
            self.model = RandomForestRegressor(n_estimators=400, min_samples_leaf=3, max_features=0.5, n_jobs=-1, random_state=RANDOM_SEED)
            self.model.fit(x.fillna(-1), y)
        elif self.kind == "extra_trees":
            self.model = ExtraTreesRegressor(n_estimators=400, min_samples_leaf=3, max_features=0.7, n_jobs=-1, random_state=RANDOM_SEED)
            self.model.fit(x.fillna(-1), y)
        elif self.kind == "hist_gb":
            categorical = [c in CATEGORICAL_FEATURES and c not in HIGH_CARDINALITY for c in x.columns]
            self.model = HistGradientBoostingRegressor(
                loss="absolute_error", learning_rate=0.05, max_iter=600, max_leaf_nodes=31,
                l2_regularization=1.0, categorical_features=categorical, early_stopping=True,
                validation_fraction=0.15, n_iter_no_change=40, random_state=RANDOM_SEED,
            )
            self.model.fit(x.replace({c: {-1: np.nan} for c in CATEGORICAL_FEATURES}), y)
        else:
            raise ValueError(self.kind)
        return self

    def predict(self, frame):
        if self.kind == "ridge":
            return np.expm1(self.model.predict(self.encoder.transform(frame)))
        x = self.encoder.transform(frame)
        if self.kind == "hist_gb":
            return self.model.predict(x.replace({c: {-1: np.nan} for c in CATEGORICAL_FEATURES}))
        return self.model.predict(x.fillna(-1))


class _XGBoost:
    def __init__(self):
        self.best_iteration: int | None = None

    def fit(self, train, validation=None):
        import xgboost as xgb

        self.encoder = _OrdinalEncoder().fit(train)
        params = dict(
            objective="reg:absoluteerror", learning_rate=0.04, max_depth=6, subsample=0.85,
            colsample_bytree=0.8, min_child_weight=3, reg_lambda=6.0, tree_method="hist",
            enable_categorical=True, max_cat_to_onehot=1, random_state=RANDOM_SEED, n_jobs=1,
        )
        x = self.encoder.transform(train, as_category=True)
        y = train[TARGET].astype(float)
        if validation is not None:
            self.model = xgb.XGBRegressor(n_estimators=1500, early_stopping_rounds=70, **params)
            self.model.fit(x, y, eval_set=[(self.encoder.transform(validation, as_category=True), validation[TARGET].astype(float))], verbose=False)
            self.best_iteration = int(self.model.best_iteration)
        else:
            self.model = xgb.XGBRegressor(n_estimators=(self.best_iteration or 600) + 1, **params)
            self.model.fit(x, y, verbose=False)
        return self

    def predict(self, frame):
        return self.model.predict(self.encoder.transform(frame, as_category=True))


class _LightGBM:
    def __init__(self):
        self.best_iteration: int | None = None

    def fit(self, train, validation=None):
        import lightgbm as lgb

        self.encoder = _OrdinalEncoder().fit(train)
        params = dict(
            objective="l1", learning_rate=0.04, num_leaves=31, min_child_samples=20,
            subsample=0.85, subsample_freq=1, colsample_bytree=0.8, reg_lambda=6.0,
            cat_smooth=20, random_state=RANDOM_SEED, n_jobs=1, verbose=-1,
        )
        x = self.encoder.transform(train, as_category=True)
        y = train[TARGET].astype(float)
        if validation is not None:
            self.model = lgb.LGBMRegressor(n_estimators=1500, **params)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # eval_set rename in newer LightGBM
                self.model.fit(
                    x, y, eval_set=[(self.encoder.transform(validation, as_category=True), validation[TARGET].astype(float))],
                    callbacks=[lgb.early_stopping(70, verbose=False)],
                )
            self.best_iteration = int(self.model.best_iteration_ or 600)
        else:
            self.model = lgb.LGBMRegressor(n_estimators=self.best_iteration or 600, **params)
            self.model.fit(x, y)
        return self

    def predict(self, frame):
        return self.model.predict(self.encoder.transform(frame, as_category=True))


def _available(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


def model_catalogue() -> list[ModelSpec]:
    """Every candidate model. XGBoost / LightGBM are skipped if not installed."""
    specs = [
        ModelSpec(BASELINE_MODEL, "Rule", "Median historical bookings for the same term and suite. The reference ML has to beat.", _Baseline),
        ModelSpec("Ridge regression (log target)", "Linear", "Regularised linear model on log(bookings) with one-hot categories. Transparent, smooth term effect.", lambda: _Sklearn("ridge")),
        ModelSpec("Random Forest", "Bagged trees", "400 averaged decision trees. Robust, little tuning, cannot extrapolate beyond seen values.", lambda: _Sklearn("random_forest")),
        ModelSpec("Extra Trees", "Bagged trees", "Like Random Forest with random split points. Lower variance, often smoother.", lambda: _Sklearn("extra_trees")),
        ModelSpec("HistGradientBoosting", "Boosted trees", "scikit-learn's LightGBM-style booster with MAE loss and native categories.", lambda: _Sklearn("hist_gb")),
    ]
    if _available("xgboost"):
        specs.append(ModelSpec("XGBoost", "Boosted trees", "Gradient boosting with MAE objective, native categoricals, early stopping on validation.", _XGBoost, True))
    if _available("lightgbm"):
        specs.append(ModelSpec("LightGBM", "Boosted trees", "Leaf-wise gradient boosting with L1 objective and native categoricals.", _LightGBM, True))
    if _available("catboost"):
        specs.append(ModelSpec(CURRENT_MODEL, "Boosted trees", "The pipeline's production model: CatBoost, MAE loss, same settings as model.py.", _CatBoost, True))
        specs.append(ModelSpec(
            "CatBoost (annualized target × term)", "Boosted trees",
            "Same CatBoost, but learns bookings ÷ term years (the annual value) and multiplies back by term years.",
            lambda: _CatBoost(annualized_target=True), True,
        ))
    return specs


# --------------------------------------------------------------------------
# Evaluation helpers
# --------------------------------------------------------------------------
def _clip(values) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), 0, None)


def _with_term(frame: pd.DataFrame, term: str) -> pd.DataFrame:
    scenario = frame.copy()
    scenario["TERM_BUCKET"] = term
    return scenario


def scenario_check(model, test: pd.DataFrame, train: pd.DataFrame) -> dict[str, float]:
    """Re-score test deals as 1/3/5-year and test whether the response is sensible."""
    p1, p3, p5 = (_clip(model.predict(_with_term(test, term))) for term in TERMS)
    safe3 = np.where(p3 > 0, p3, np.nan)
    annual3 = safe3 / 3
    observed = (train[TARGET] / train["TERM_BUCKET"].map(TERM_YEARS)).groupby(train["TERM_BUCKET"]).median()
    # Default app scenario: 100% flexible adoption, 50% five-year share.
    mixed = 0.5 * p1 + 0.5 * p5
    return {
        "monotonic_pct": float(np.mean((p1 < p3) & (p3 < p5)) * 100),
        "flat_term_pct": float(np.mean(np.isclose(p1, p3, rtol=0.01) | np.isclose(p3, p5, rtol=0.01)) * 100),
        "annual_ratio_1y_vs_3y": float(np.nanmedian(p1 / annual3)),
        "annual_ratio_5y_vs_3y": float(np.nanmedian((p5 / 5) / annual3)),
        "observed_ratio_1y_vs_3y": float(observed.get("1 Year", np.nan) / observed.get("3 Year", np.nan)),
        "observed_ratio_5y_vs_3y": float(observed.get("5 Year", np.nan) / observed.get("3 Year", np.nan)),
        "default_scenario_change_pct": float((mixed.sum() - p3.sum()) / p3.sum() * 100) if p3.sum() else np.nan,
    }


# Slider defaults on the main app page.
DEFAULT_ADOPTION_PCT, DEFAULT_FIVE_YEAR_SHARE_PCT, DEFAULT_DISCOUNT_RATE_PCT = 100, 50, 8


def _lens_outputs(term_bookings: dict[int, np.ndarray], frame: pd.DataFrame) -> dict[str, float]:
    """Run the app's own scenario.py maths and return the three-lens numbers."""
    from .scenario import annual_coverage, breakeven_share, calculate_scenario, sensitivity, summarise

    scored = frame[["SUBSCRIPTION_REFERENCE_ID"]].reset_index(drop=True).copy()
    for years, values in term_bookings.items():
        scored[f"bookings_{years}y"] = values
        # Lens check uses point estimates only; ranges come from the quantile models.
        scored[f"bookings_{years}y_p10"] = values
        scored[f"bookings_{years}y_p90"] = values
    details = calculate_scenario(scored, DEFAULT_ADOPTION_PCT, DEFAULT_FIVE_YEAR_SHARE_PCT, DEFAULT_DISCOUNT_RATE_PCT)
    coverage = annual_coverage(details, DEFAULT_ADOPTION_PCT, DEFAULT_FIVE_YEAR_SHARE_PCT)
    summary = summarise(details, coverage)
    breakeven = breakeven_share(sensitivity(scored, DEFAULT_ADOPTION_PCT, DEFAULT_DISCOUNT_RATE_PCT))
    return {
        # Lens 1 · Growth
        "growth_commitment_change_pct": summary["commitment_change_pct"],
        "growth_npv_change_pct": summary["npv_change"] / summary["baseline_npv"] * 100 if summary["baseline_npv"] else np.nan,
        "growth_npv_breakeven_5y_share_pct": np.nan if breakeven is None else breakeven,
        # Lens 2 · Customer value (lever-driven proxy: identical for every model)
        "customer_avg_term_years": summary["average_term_years"],
        # Lens 3 · Simplicity with scale
        "simplicity_coverage_cv_pct": summary["scenario_coverage_cv_pct"],
        "simplicity_coverage_cv_change_pts": summary["scenario_coverage_cv_pct"] - summary["baseline_coverage_cv_pct"],
        "simplicity_renewal_points_change": summary["scenario_renewal_points"] - summary["baseline_renewal_points"],
    }


def lens_check(model, test: pd.DataFrame) -> dict[str, float]:
    """Three business lenses (PPT slide 3) computed from this model's 1/3/5-year scores."""
    return _lens_outputs({TERM_YEARS[t]: _clip(model.predict(_with_term(test, t))) for t in TERMS}, test)


def synthetic_truth_lenses(test: pd.DataFrame) -> dict[str, float]:
    """Exact lens values on synthetic data, using the generator's known term factors."""
    own_factor = test["TERM_BUCKET"].map(TERM_YEARS).map(SYNTHETIC_TERM_ANNUAL_FACTOR).to_numpy(float)
    base_annual = test["ANNUALIZED_NET_PRICE_USD_AMT"].to_numpy(float) / own_factor
    return _lens_outputs({t: base_annual * SYNTHETIC_TERM_ANNUAL_FACTOR[t] * t for t in (1, 3, 5)}, test)


def backtest_folds(train_validation: pd.DataFrame, n_folds: int = 3) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Expanding-window folds over distinct start dates (never touches test)."""
    data = train_validation.dropna(subset=["TERM_START_DATE"]).sort_values("TERM_START_DATE")
    dates = np.array(sorted(data["TERM_START_DATE"].unique()))
    if len(dates) < n_folds + 2:
        return []
    edges = np.linspace(0.5, 1.0, n_folds + 1)
    folds = []
    for start, end in zip(edges[:-1], edges[1:]):
        cut = pd.Timestamp(dates[int(len(dates) * start) - 1])
        stop = pd.Timestamp(dates[min(int(len(dates) * end), len(dates)) - 1])
        fit_part = data[data.TERM_START_DATE <= cut]
        eval_part = data[(data.TERM_START_DATE > cut) & (data.TERM_START_DATE <= stop)]
        if len(fit_part) and len(eval_part):
            folds.append((fit_part, eval_part))
    return folds


@dataclass
class ComparisonResult:
    leaderboard: pd.DataFrame
    metrics_by_term: pd.DataFrame
    backtest: pd.DataFrame
    scenario: pd.DataFrame
    predictions: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)
    lenses: pd.DataFrame = field(default_factory=pd.DataFrame)


def _recommend(board: pd.DataFrame) -> tuple[str, str]:
    safe = board[board.scenario_safe & (board.model != BASELINE_MODEL)]
    pool = safe if not safe.empty else board[board.model != BASELINE_MODEL]
    best = pool.sort_values(["test_wape_pct", "backtest_wape_std"]).iloc[0]
    current = board[board.model == CURRENT_MODEL]
    if current.empty:
        return best.model, "CatBoost not installed; best scenario-safe model by test WAPE."
    cur = current.iloc[0]
    gap = cur.test_wape_pct - best.test_wape_pct
    # "Noise" = how much CatBoost's own WAPE moves between backtest folds.
    noise = cur.backtest_wape_std if pd.notna(cur.backtest_wape_std) and cur.backtest_wape_std > 0 else 1.0
    if best.model == CURRENT_MODEL:
        return CURRENT_MODEL, "The current CatBoost model has the lowest test WAPE among scenario-safe models."
    if gap < noise:
        return CURRENT_MODEL, (
            f"Keep CatBoost. {best.model} is {gap:.2f} WAPE points better on the test period, "
            f"which is smaller than CatBoost's own fold-to-fold spread ({noise:.2f} points), so the gain is not reliable. "
            "Re-run on Cisco data before switching.")
    return best.model, (f"{best.model} beats the current CatBoost by {gap:.2f} WAPE points on the test period "
                        f"(more than CatBoost's {noise:.2f}-point fold spread) and passes the scenario check.")


def different_models(
    frame: pd.DataFrame,
    models: list[str] | None = None,
    n_backtest_folds: int = 3,
    progress: Callable[[str], None] | None = None,
) -> ComparisonResult:
    """Train and compare several model families on the Cluster C bookings target.

    Parameters
    ----------
    frame:
        Prepared features (``data/prepared/term_model_features.csv``).
    models:
        Optional subset of model names from ``model_catalogue()``; default all.
    n_backtest_folds:
        Expanding-window folds for the stability check; 0 disables it.
    progress:
        Optional callback receiving status text (used by the Streamlit page).
    """
    say = progress or (lambda message: LOGGER.info(message))
    specs = [s for s in model_catalogue() if models is None or s.name in models]
    if not specs:
        raise ValueError("No models selected")
    splits = chronological_split(frame)
    train, validation, test = splits["train"], splits["validation"], splits["test"]
    train_validation = pd.concat([train, validation])
    folds = backtest_folds(train_validation, n_backtest_folds) if n_backtest_folds else []

    board_rows, term_rows, cv_rows, scenario_rows, lens_rows, prediction_frames = [], [], [], [], [], []
    for spec in specs:
        say(f"Training {spec.name}")
        model = spec.builder()
        started = time.perf_counter()
        model.fit(train, validation if spec.uses_validation else None)
        fit_seconds = time.perf_counter() - started

        split_metrics = {}
        for split_name, part in (("validation", validation), ("test", test)):
            started = time.perf_counter()
            predicted = _clip(model.predict(part))
            predict_ms = (time.perf_counter() - started) / max(len(part), 1) * 1000
            actual = part[TARGET].to_numpy(float)
            split_metrics[split_name] = regression_metrics(actual, predicted)
            for term in TERMS:
                mask = part["TERM_BUCKET"].eq(term).to_numpy()
                term_rows.append({"model": spec.name, "split": split_name, "term_bucket": term, "rows": int(mask.sum()), **regression_metrics(actual[mask], predicted[mask])})
            if split_name == "test":
                prediction_frames.append(pd.DataFrame({
                    "model": spec.name, "SUBSCRIPTION_REFERENCE_ID": part["SUBSCRIPTION_REFERENCE_ID"].to_numpy(),
                    "TERM_START_DATE": part["TERM_START_DATE"].to_numpy(), "TERM_BUCKET": part["TERM_BUCKET"].to_numpy(),
                    "SUITE_NAME": part["SUITE_NAME"].to_numpy(), "SCMS": part["SCMS"].to_numpy(),
                    "actual": actual, "predicted": predicted,
                }))

        check = scenario_check(model, test, train)
        if is_synthetic_dataset():
            check["synthetic_true_change_pct"] = SYNTHETIC_TRUE_DEFAULT_CHANGE_PCT
            check["scenario_error_vs_truth_pts"] = check["default_scenario_change_pct"] - SYNTHETIC_TRUE_DEFAULT_CHANGE_PCT
        scenario_rows.append({"model": spec.name, **check})
        lens_rows.append({"model": spec.name, **lens_check(model, test)})

        fold_wapes = []
        for index, (fit_part, eval_part) in enumerate(folds, start=1):
            say(f"Backtesting {spec.name} (fold {index}/{len(folds)})")
            fold_model = spec.builder()
            # Reuse the early-stopped iteration count so folds need no extra holdout.
            if hasattr(model, "best_iteration"):
                fold_model.best_iteration = model.best_iteration
            fold_model.fit(fit_part, None)
            fold_metrics = regression_metrics(eval_part[TARGET].to_numpy(float), _clip(fold_model.predict(eval_part)))
            fold_wapes.append(fold_metrics["wape_pct"])
            cv_rows.append({"model": spec.name, "fold": index, "fit_rows": len(fit_part), "eval_rows": len(eval_part),
                            "eval_start": eval_part.TERM_START_DATE.min(), "eval_end": eval_part.TERM_START_DATE.max(), **fold_metrics})

        test_m, val_m = split_metrics["test"], split_metrics["validation"]
        board_rows.append({
            "model": spec.name, "family": spec.family, "description": spec.description,
            **{f"test_{k}": v for k, v in test_m.items()},
            "validation_wape_pct": val_m["wape_pct"],
            "backtest_wape_mean": float(np.mean(fold_wapes)) if fold_wapes else np.nan,
            "backtest_wape_std": float(np.std(fold_wapes)) if fold_wapes else np.nan,
            "fit_seconds": fit_seconds, "predict_ms_per_row": predict_ms,
            "monotonic_pct": check["monotonic_pct"],
            "default_scenario_change_pct": check["default_scenario_change_pct"],
            "scenario_safe": check["monotonic_pct"] >= MONOTONIC_THRESHOLD_PCT,
        })
        LOGGER.info("%s: test WAPE %.2f%%, monotonic %.1f%%", spec.name, test_m["wape_pct"], check["monotonic_pct"])

    board = pd.DataFrame(board_rows).sort_values("test_wape_pct", ignore_index=True)
    board.insert(0, "rank", range(1, len(board) + 1))
    baseline = board.loc[board.model == BASELINE_MODEL, "test_wape_pct"]
    board["wape_gain_vs_baseline_pct"] = (
        (baseline.iloc[0] - board.test_wape_pct) / baseline.iloc[0] * 100 if not baseline.empty else np.nan
    )
    lenses = pd.DataFrame(lens_rows)
    if is_synthetic_dataset():
        truth = synthetic_truth_lenses(test)
        for column in ("growth_commitment_change_pct", "growth_npv_change_pct", "simplicity_coverage_cv_pct"):
            lenses[column.replace("_pct", "") + "_error_vs_truth_pts"] = lenses[column] - truth[column]
        lenses = pd.concat([lenses, pd.DataFrame([{"model": SYNTHETIC_TRUTH_LABEL, **truth}])], ignore_index=True)
    recommended, reason = _recommend(board)
    board["recommended"] = board.model.eq(recommended)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset_version": SOURCE_DATASET_VERSION,
        "target": TARGET,
        "target_definition": "Total committed bookings over the whole term (USD) = annualized net price × term years",
        "split_rows": {name: len(part) for name, part in splits.items()},
        "split_dates": {name: [str(part.TERM_START_DATE.min().date()), str(part.TERM_START_DATE.max().date())] for name, part in splits.items()},
        "primary_metric": "test_wape_pct",
        "scenario_safe_rule": f"monotonic_pct >= {MONOTONIC_THRESHOLD_PCT}",
        "backtest_folds": len(folds),
        "models": [s.name for s in specs],
        "recommended_model": recommended,
        "recommendation_reason": reason,
        "current_model": CURRENT_MODEL,
        "synthetic_dataset": is_synthetic_dataset(),
        "synthetic_true_default_change_pct": SYNTHETIC_TRUE_DEFAULT_CHANGE_PCT if is_synthetic_dataset() else None,
    }
    return ComparisonResult(board, pd.DataFrame(term_rows), pd.DataFrame(cv_rows), pd.DataFrame(scenario_rows),
                            pd.concat(prediction_frames, ignore_index=True), metadata, lenses)


def save_comparison(result: ComparisonResult, output_dir: Path = COMPARISON_DIR) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "leaderboard": output_dir / "leaderboard.csv", "metrics_by_term": output_dir / "metrics_by_term.csv",
        "backtest": output_dir / "backtest.csv", "scenario": output_dir / "scenario_check.csv",
        "predictions": output_dir / "test_predictions.csv", "metadata": output_dir / "metadata.json",
        "lenses": output_dir / "business_lenses.csv",
    }
    result.lenses.to_csv(paths["lenses"], index=False)
    result.leaderboard.to_csv(paths["leaderboard"], index=False)
    result.metrics_by_term.to_csv(paths["metrics_by_term"], index=False)
    result.backtest.to_csv(paths["backtest"], index=False, date_format="%Y-%m-%d")
    result.scenario.to_csv(paths["scenario"], index=False)
    result.predictions.to_csv(paths["predictions"], index=False, date_format="%Y-%m-%d")
    paths["metadata"].write_text(json.dumps(result.metadata, indent=2) + "\n", encoding="utf-8")
    return paths


def load_comparison(output_dir: Path = COMPARISON_DIR) -> ComparisonResult:
    read = lambda name: pd.read_csv(output_dir / name)
    return ComparisonResult(
        read("leaderboard.csv"), read("metrics_by_term.csv"),
        read("backtest.csv") if (output_dir / "backtest.csv").stat().st_size > 2 else pd.DataFrame(),
        read("scenario_check.csv"), read("test_predictions.csv"),
        json.loads((output_dir / "metadata.json").read_text(encoding="utf-8")),
        read("business_lenses.csv") if (output_dir / "business_lenses.csv").exists() else pd.DataFrame(),
    )
