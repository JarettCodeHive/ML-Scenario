"""Compare different regression models for the Cluster C bookings target.

WHAT THIS FILE ANSWERS
    "Is CatBoost the right model for the term-flexibility what-if, or would
    another model do better?"

WHAT IS BEING PREDICTED (the target)
    BOOKINGS_NET_PRICE_USD_AMT = total committed bookings for the whole
    contract term, in USD. In this data it equals
    ANNUALIZED_NET_PRICE_USD_AMT x term years, so a 5-year deal books
    roughly 5x its annual value.

HOW THE COMPARISON WORKS (big picture)
    ``different_models()`` trains every candidate model on exactly the same
    chronological train / validation / test split that the production CatBoost
    uses (oldest deals = train, newer = validation, newest = test), then
    scores each model four ways:

    1. Accuracy on the untouched test period: MAE, RMSE, WAPE, bias, R²,
       MAPE, share of deals predicted within ±20%, overall and per term.
    2. Stability: an expanding-window backtest over train + validation
       (test is never touched), reported as mean and spread of WAPE.
    3. Scenario fitness: every test deal is re-scored as 1, 3 and 5 years,
       exactly the way the what-if engine does. A model that is accurate but
       does not react sensibly to TERM_BUCKET is not safe for the scenario.
    4. The three business lenses from the Cisco PPT (Growth, Customer value,
       Simplicity), calculated with the app's own scenario.py maths.

HOW TO READ THIS FILE (top to bottom)
    Constants            -> names, thresholds, the known synthetic "truth"
    regression_metrics() -> the accuracy formulas
    _OrdinalEncoder / _OneHotEncoder -> turn text columns into numbers for
                            libraries that cannot read text (CatBoost can)
    _Baseline, _CatBoost, _Sklearn, _XGBoost, _LightGBM
                         -> one small wrapper per library, all with the same
                            two methods: fit(train, validation) and predict(frame)
    model_catalogue()    -> the list of models to compare
    scenario_check(), lens_check(), backtest_folds()
                         -> the extra checks beyond plain accuracy
    _recommend()         -> picks the model to recommend, with a reason
    different_models()   -> THE MAIN FUNCTION: runs everything above
    save_comparison() / load_comparison() -> write/read the CSV results

This module is evaluation only. It does NOT replace the CatBoost models that
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

# Re-use the pipeline's own building blocks so every model sees the same data:
# - chronological_split: the same train/validation/test dates as model.py
# - matrix: the same feature table (missing categories -> "Unknown", numbers coerced)
# - FEATURES / CATEGORICAL_FEATURES / TARGET: the same columns as production
from .config import DATA_DIR, RANDOM_SEED, SOURCE_DATASET_VERSION
from .model import TrainingConfig, chronological_split, matrix
from .preparation import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET

LOGGER = logging.getLogger(__name__)

# ==========================================================================
# CONSTANTS
# ==========================================================================
# Where the comparison results (CSV + JSON) are written.
COMPARISON_DIR = DATA_DIR / "models" / "model_comparison"

# The three term options the what-if engine switches between.
TERMS = ("1 Year", "3 Year", "5 Year")
TERM_YEARS = {"1 Year": 1, "3 Year": 3, "5 Year": 5}

# Display names used on the leaderboard and dashboard.
CURRENT_MODEL = "CatBoost (current)"
BASELINE_MODEL = "Term × suite median (baseline)"

# Identifiers with hundreds of distinct values (~700 customers). CatBoost
# handles these well, but they make a one-hot linear model overfit, so the
# linear model leaves them out, and XGBoost/LightGBM treat them as plain
# integer codes instead of categories (see _OrdinalEncoder.transform).
HIGH_CARDINALITY = {"CAV_END_CUSTOMER_BU_ID", "UCD_C4_ACCOUNT_ID"}

# Dollar-valued history features. They are very skewed, so the linear model
# uses log(1 + value) and adds a 0/1 flag for "customer has no history yet".
MONEY_FEATURES = ["CUSTOMER_PRIOR_AVG_ANNUALIZED_PRICE", "CUSTOMER_PRIOR_AVG_BOOKINGS"]

# --- Known truth on synthetic data ----------------------------------------
# The synthetic generator (synthetic.py) multiplies a deal's annual value by
# these factors for 1/3/5-year terms. Because we wrote the generator, we know
# the TRUE effect of the default scenario (100% flexible, 50% five-year):
#   (0.5*1*0.94 + 0.5*5*1.075 - 3*1.0) / (3*1.0) = +5.25%
# That lets us grade each model's what-if answer, not just its accuracy.
# Real Cisco data has no known truth; this check is only shown for synthetic runs.
SYNTHETIC_TERM_ANNUAL_FACTOR = {1: 0.94, 3: 1.00, 5: 1.075}
SYNTHETIC_TRUE_DEFAULT_CHANGE_PCT = (
    (0.5 * 1 * SYNTHETIC_TERM_ANNUAL_FACTOR[1] + 0.5 * 5 * SYNTHETIC_TERM_ANNUAL_FACTOR[5] - 3 * SYNTHETIC_TERM_ANNUAL_FACTOR[3])
    / (3 * SYNTHETIC_TERM_ANNUAL_FACTOR[3]) * 100
)


# Row label for the "known truth" line in the business-lenses table.
SYNTHETIC_TRUTH_LABEL = "Known truth (synthetic data)"


def _term_years(term_bucket: pd.Series) -> pd.Series:
    """'1 Year' / '3 Year' / '5 Year' -> 1.0 / 3.0 / 5.0.

    Works whether TERM_BUCKET is plain text (CSV) or a pandas category
    (the Parquet file written by prepare_data.py), where .map() would
    otherwise return a category column that cannot be used in arithmetic.
    """
    return term_bucket.astype("string").map(TERM_YEARS).astype(float)


def is_synthetic_dataset(version: str = SOURCE_DATASET_VERSION) -> bool:
    """True when the data is the generated sample (its version label says 'synthetic')."""
    return "synthetic" in str(version).lower()


# A model is flagged "scenario-unsafe" when fewer test deals than this keep
# the expected order bookings(1y) < bookings(3y) < bookings(5y). A model that
# says a 1-year deal books more than a 5-year deal would give nonsense what-ifs.
MONOTONIC_THRESHOLD_PCT = 95.0


# ==========================================================================
# METRICS
# ==========================================================================
def regression_metrics(actual, predicted) -> dict[str, float]:
    """Accuracy metrics used on the leaderboard (WAPE is the primary one).

    Example: actual = [100, 200], predicted = [110, 180]
      errors = [+10, -20]
      MAE  = (10 + 20) / 2           = 15        (average $ miss per deal)
      WAPE = (10 + 20) / (100 + 200) = 10%       (miss as % of all bookings)
      Bias = (+10 - 20) / 300        = -3.3%     (negative = under-predicts)
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    keys = ("mae", "rmse", "wape_pct", "bias_pct", "r2", "mape_pct", "within_20pct")
    # An empty slice (e.g. no 5-year deals in a split) gives NaN, not a crash.
    if actual.size == 0:
        return {key: np.nan for key in keys}
    error = predicted - actual                              # + = over-predicted, - = under-predicted
    abs_total = float(np.abs(actual).sum())                 # total bookings (WAPE denominator)
    total = float(actual.sum())                             # signed total (bias denominator)
    ss_tot = float(np.square(actual - actual.mean()).sum())  # total variation (R² denominator)
    nonzero = actual != 0                                   # per-deal % error is undefined for $0 deals
    ape = np.abs(error[nonzero] / actual[nonzero])          # absolute % error per deal
    return {
        # Average dollar miss per deal.
        "mae": float(np.abs(error).mean()),
        # Like MAE but squares errors first, so a few big misses weigh more.
        "rmse": float(np.sqrt(np.square(error).mean())),
        # PRIMARY METRIC: total $ missed / total $ booked. Big deals count more,
        # which matches a portfolio-level what-if.
        "wape_pct": float(np.abs(error).sum() / abs_total * 100) if abs_total else np.nan,
        # Does the model lean high (+) or low (-) overall?
        "bias_pct": float(error.sum() / total * 100) if total else np.nan,
        # Share of the variation in bookings the model explains (1.0 = perfect).
        "r2": float(1 - np.square(error).sum() / ss_tot) if ss_tot else np.nan,
        # Average % miss per deal; a $20k deal counts as much as a $2M deal.
        "mape_pct": float(ape.mean() * 100) if ape.size else np.nan,
        # Share of deals predicted within ±20% of the real value.
        "within_20pct": float((ape <= 0.20).mean() * 100) if ape.size else np.nan,
    }


# ==========================================================================
# FEATURE ENCODINGS (for the libraries that cannot read text columns)
# ==========================================================================
# CatBoost reads text categories like "Security" directly. scikit-learn,
# XGBoost and LightGBM need numbers, so these two helpers convert the same
# feature table into numeric form. Both LEARN their mapping on the training
# rows only (fit) and then APPLY it unchanged to validation/test (transform),
# so no information from the future leaks into training.
class _OrdinalEncoder:
    """Turn each category into an integer code, e.g. Security -> 3. Unseen -> -1.

    Used by the tree models (Random Forest, Extra Trees, HistGradientBoosting,
    XGBoost, LightGBM). Trees only split on "code <= x", so arbitrary integer
    codes work fine for them.
    """

    def __init__(self) -> None:
        # column name -> {category text: integer code}
        self.maps: dict[str, dict[str, int]] = {}

    def fit(self, frame: pd.DataFrame) -> "_OrdinalEncoder":
        base = matrix(frame)
        for column in CATEGORICAL_FEATURES:
            # Sorted so the codes are deterministic run to run.
            levels = sorted(base[column].unique())
            self.maps[column] = {level: code for code, level in enumerate(levels)}
        return self

    def transform(self, frame: pd.DataFrame, as_category: bool = False) -> pd.DataFrame:
        """as_category=True is for XGBoost/LightGBM, which can split on categories natively."""
        out = matrix(frame)
        for column in CATEGORICAL_FEATURES:
            # A category never seen in training (e.g. a brand-new customer) becomes -1.
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
        # term length (1, 3, 5) is clearer for every model, so add it explicitly.
        out["TERM_YEARS_NUM"] = _term_years(matrix(frame)["TERM_BUCKET"])
        return out


class _OneHotEncoder:
    """Scaled numbers + one 0/1 column per category, for the linear (Ridge) model.

    A linear model would wrongly read integer codes as "Security (3) is three
    times Applications (1)", so each category gets its own 0/1 column instead.
    All columns are then standardised (mean 0, std 1) so the Ridge penalty
    treats them equally.
    """

    def __init__(self) -> None:
        self.categories: dict[str, list[str]] = {}   # column -> categories seen in training
        self.columns: list[str] = []                 # final column order learned in fit
        self.medians: dict[str, float] = {}          # fill-in values for missing numbers
        self.mean: pd.Series | None = None           # for standardising
        self.std: pd.Series | None = None

    def _numeric(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Numeric features, with dollar columns logged and a 'missing history' flag added."""
        base = matrix(frame)
        numeric = base[[c for c in FEATURES if c in NUMERIC_FEATURES]].astype(float).copy()
        for column in MONEY_FEATURES:
            # First deal for a customer has no history -> flag it before filling the gap.
            numeric[column + "_MISSING"] = numeric[column].isna().astype(float)
            numeric[column] = np.log1p(numeric[column].clip(lower=0))
        return numeric

    def fit(self, frame: pd.DataFrame) -> "_OneHotEncoder":
        base = matrix(frame)
        for column in CATEGORICAL_FEATURES:
            if column not in HIGH_CARDINALITY:   # ~700-level IDs are skipped (would overfit)
                self.categories[column] = sorted(base[column].unique())
        numeric = self._numeric(frame)
        self.medians = numeric.median().fillna(0).to_dict()
        encoded = self._encode(frame, fitting=True)
        self.columns = list(encoded.columns)
        return self

    def _encode(self, frame: pd.DataFrame, fitting: bool = False) -> pd.DataFrame:
        base = matrix(frame)
        numeric = self._numeric(frame).fillna(self.medians)
        # One 0/1 column per category value, e.g. SUITE_NAME_Security = 1.
        dummies = [
            pd.get_dummies(pd.Categorical(base[col], categories=levels), prefix=col, dtype=float).set_index(base.index)
            for col, levels in self.categories.items()
        ]
        encoded = pd.concat([numeric, *dummies], axis=1)
        if fitting:
            # Scaling is learned on training rows only, then reused for other splits.
            self.mean = encoded.mean()
            self.std = encoded.std().replace(0, 1).fillna(1)
        return (encoded - self.mean) / self.std

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        # reindex guarantees the same columns in the same order as training.
        return self._encode(frame).reindex(columns=self.columns, fill_value=0.0)


# ==========================================================================
# MODEL WRAPPERS
# ==========================================================================
# Every library has its own API. Each wrapper below hides that behind the
# same two methods, so different_models() can treat all models identically:
#   fit(train, validation)  -> learns from train; validation (if given) is
#                              used only to decide when to stop boosting
#   predict(frame)          -> returns predicted bookings in USD
@dataclass
class ModelSpec:
    """One entry in the comparison: what the model is called and how to build it."""
    name: str                       # label on the leaderboard
    family: str                     # Rule / Linear / Bagged trees / Boosted trees
    description: str                # one-line plain-English explanation for the dashboard
    builder: Callable[[], Any]      # function that returns a fresh, untrained wrapper
    uses_validation: bool = False   # True = pass validation rows for early stopping


class _Baseline:
    """No ML: predict the median bookings of past deals with the same term and suite.

    This is the 'is ML worth it?' reference. Every real model must beat it.
    """

    def fit(self, train, validation=None):
        # e.g. median bookings for (3 Year, Security) across training deals
        self.medians = train.groupby(["TERM_BUCKET", "SUITE_NAME"], observed=True)[TARGET].median()
        # Fallback for a term/suite pair never seen in training.
        self.global_median = float(train[TARGET].median())
        return self

    def predict(self, frame):
        return np.array([
            self.medians.get((row.TERM_BUCKET, row.SUITE_NAME), self.global_median)
            for row in frame[["TERM_BUCKET", "SUITE_NAME"]].itertuples()
        ], dtype=float)


class _CatBoost:
    """Same CatBoost settings as model.py, so 'CatBoost (current)' matches production.

    annualized_target=True (production since model v2): the model learns
    bookings / term years (the ANNUAL value) and multiplies back by the term at
    prediction time. Because total = annual x years exactly, this builds that
    rule into the model instead of making it learn it.
    annualized_target=False is the previous production setup (learns total
    bookings directly), kept so the comparison shows before vs after.
    """

    def __init__(self, annualized_target: bool = True, config: TrainingConfig = TrainingConfig()):
        self.annualized_target = annualized_target
        self.config = config                      # depth, learning rate, etc. from model.py
        self.best_iteration: int | None = None    # set by early stopping; reused in backtest folds

    def _label(self, frame):
        """What the model learns: total bookings, or bookings per year for the variant."""
        label = frame[TARGET].astype(float)
        return label / _term_years(frame["TERM_BUCKET"]) if self.annualized_target else label

    def fit(self, train, validation=None):
        from catboost import CatBoostRegressor, Pool

        cfg = self.config
        # With validation rows: train up to the max and let early stopping pick the best point.
        # Without (backtest folds): reuse the number of trees found earlier.
        iterations = cfg.iterations if validation is not None or self.best_iteration is None else self.best_iteration + 1
        self.model = CatBoostRegressor(
            iterations=iterations, depth=cfg.depth, learning_rate=cfg.learning_rate,
            l2_leaf_reg=cfg.l2_leaf_reg, loss_function="MAE", random_seed=cfg.random_seed,
            allow_writing_files=False, thread_count=cfg.thread_count, verbose=False,
            # Same category-combination limits as production (model._new_model).
            max_ctr_complexity=cfg.max_ctr_complexity, ctr_leaf_count_limit=cfg.ctr_leaf_count_limit,
        )
        # Pool = CatBoost's data container; cat_features tells it which columns are text.
        pool = Pool(matrix(train), label=self._label(train), cat_features=CATEGORICAL_FEATURES)
        if validation is not None:
            eval_pool = Pool(matrix(validation), label=self._label(validation), cat_features=CATEGORICAL_FEATURES)
            # Stop adding trees once validation error has not improved for N rounds.
            self.model.fit(pool, eval_set=eval_pool, early_stopping_rounds=cfg.early_stopping_rounds, verbose=False)
            self.best_iteration = int(self.model.get_best_iteration())
        else:
            self.model.fit(pool, verbose=False)
        return self

    def predict(self, frame):
        prediction = self.model.predict(matrix(frame))
        if self.annualized_target:
            # Annual value x term years = total bookings (same unit as every other model).
            prediction = prediction * _term_years(frame["TERM_BUCKET"]).to_numpy()
        return prediction


class _Sklearn:
    """scikit-learn models: kind = ridge | random_forest | extra_trees | hist_gb."""

    def __init__(self, kind: str):
        self.kind = kind

    def fit(self, train, validation=None):
        # Imported here so the file still loads if a library is missing.
        from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
        from sklearn.linear_model import RidgeCV

        y = train[TARGET].astype(float).to_numpy()
        if self.kind == "ridge":
            # Linear model: one-hot features, and predict log(bookings) because
            # bookings are heavily right-skewed (a few very large deals).
            # RidgeCV tries 20 penalty strengths and keeps the best one.
            self.encoder = _OneHotEncoder().fit(train)
            self.model = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(self.encoder.transform(train), np.log1p(y))
            return self
        # All tree models use integer-coded categories.
        self.encoder = _OrdinalEncoder().fit(train)
        x = self.encoder.transform(train)
        if self.kind == "random_forest":
            # 400 independent deep trees, each on a random sample, then averaged.
            self.model = RandomForestRegressor(n_estimators=400, min_samples_leaf=3, max_features=0.5, n_jobs=-1, random_state=RANDOM_SEED)
            self.model.fit(x.fillna(-1), y)   # these trees cannot take NaN -> use -1
        elif self.kind == "extra_trees":
            # Like Random Forest, but split points are chosen at random: smoother, less variance.
            self.model = ExtraTreesRegressor(n_estimators=400, min_samples_leaf=3, max_features=0.7, n_jobs=-1, random_state=RANDOM_SEED)
            self.model.fit(x.fillna(-1), y)
        elif self.kind == "hist_gb":
            # Boosting: trees built one after another, each fixing the previous errors.
            # MAE loss to match CatBoost; low-cardinality columns treated as categories.
            categorical = [c in CATEGORICAL_FEATURES and c not in HIGH_CARDINALITY for c in x.columns]
            self.model = HistGradientBoostingRegressor(
                loss="absolute_error", learning_rate=0.05, max_iter=600, max_leaf_nodes=31,
                l2_regularization=1.0, categorical_features=categorical, early_stopping=True,
                validation_fraction=0.15, n_iter_no_change=40, random_state=RANDOM_SEED,
            )
            # HistGradientBoosting expects unseen categories as NaN, not -1.
            self.model.fit(x.replace({c: {-1: np.nan} for c in CATEGORICAL_FEATURES}), y)
        else:
            raise ValueError(self.kind)
        return self

    def predict(self, frame):
        if self.kind == "ridge":
            # Undo the log: exp(prediction) - 1 gives bookings in USD again.
            return np.expm1(self.model.predict(self.encoder.transform(frame)))
        x = self.encoder.transform(frame)
        if self.kind == "hist_gb":
            return self.model.predict(x.replace({c: {-1: np.nan} for c in CATEGORICAL_FEATURES}))
        return self.model.predict(x.fillna(-1))


class _XGBoost:
    """XGBoost gradient boosting with MAE objective and early stopping on validation."""

    def __init__(self):
        self.best_iteration: int | None = None

    def fit(self, train, validation=None):
        import xgboost as xgb

        self.encoder = _OrdinalEncoder().fit(train)
        params = dict(
            objective="reg:absoluteerror",   # MAE, same as CatBoost
            learning_rate=0.04, max_depth=6,
            subsample=0.85, colsample_bytree=0.8,   # each tree sees 85% of rows / 80% of columns
            min_child_weight=3, reg_lambda=6.0,     # regularisation against overfitting
            tree_method="hist", enable_categorical=True, max_cat_to_onehot=1,
            random_state=RANDOM_SEED, n_jobs=1,
        )
        x = self.encoder.transform(train, as_category=True)
        y = train[TARGET].astype(float)
        if validation is not None:
            # Up to 1500 trees; stop when validation error stalls for 70 rounds.
            self.model = xgb.XGBRegressor(n_estimators=1500, early_stopping_rounds=70, **params)
            self.model.fit(x, y, eval_set=[(self.encoder.transform(validation, as_category=True), validation[TARGET].astype(float))], verbose=False)
            self.best_iteration = int(self.model.best_iteration)
        else:
            # Backtest folds: no validation rows, so reuse the tree count found earlier.
            self.model = xgb.XGBRegressor(n_estimators=(self.best_iteration or 600) + 1, **params)
            self.model.fit(x, y, verbose=False)
        return self

    def predict(self, frame):
        return self.model.predict(self.encoder.transform(frame, as_category=True))


class _LightGBM:
    """LightGBM leaf-wise boosting with L1 (MAE) objective and early stopping."""

    def __init__(self):
        self.best_iteration: int | None = None

    def fit(self, train, validation=None):
        import lightgbm as lgb

        self.encoder = _OrdinalEncoder().fit(train)
        params = dict(
            objective="l1",                       # MAE, same as CatBoost
            learning_rate=0.04, num_leaves=31, min_child_samples=20,
            subsample=0.85, subsample_freq=1, colsample_bytree=0.8, reg_lambda=6.0,
            cat_smooth=20,                        # smooths category statistics for rare categories
            random_state=RANDOM_SEED, n_jobs=1, verbose=-1,
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
    """True if a Python library is installed (used to skip XGBoost/LightGBM if missing)."""
    try:
        __import__(module)
        return True
    except ImportError:
        return False


def model_catalogue() -> list[ModelSpec]:
    """The list of models to compare. To add a new model, add one ModelSpec here.

    XGBoost / LightGBM / CatBoost are skipped automatically if not installed.
    """
    specs = [
        ModelSpec(BASELINE_MODEL, "Rule", "Median historical bookings for the same term and suite. The reference ML has to beat.", _Baseline),
        ModelSpec("Ridge regression (log target)", "Linear", "Regularised linear model on log(bookings) with one-hot categories. Transparent, smooth term effect.", lambda: _Sklearn("ridge")),
        ModelSpec("Random Forest", "Bagged trees", "400 averaged decision trees. Robust, little tuning, cannot extrapolate beyond seen values.", lambda: _Sklearn("random_forest")),
        ModelSpec("Extra Trees", "Bagged trees", "Like Random Forest with random split points. Lower variance, often smoother.", lambda: _Sklearn("extra_trees")),
        ModelSpec("HistGradientBoosting", "Boosted trees", "scikit-learn's LightGBM-style booster with MAE loss and native categories.", lambda: _Sklearn("hist_gb")),
    ]
    # The last argument True = this model receives the validation rows for early stopping.
    if _available("xgboost"):
        specs.append(ModelSpec("XGBoost", "Boosted trees", "Gradient boosting with MAE objective, native categoricals, early stopping on validation.", _XGBoost, True))
    if _available("lightgbm"):
        specs.append(ModelSpec("LightGBM", "Boosted trees", "Leaf-wise gradient boosting with L1 objective and native categoricals.", _LightGBM, True))
    if _available("catboost"):
        specs.append(ModelSpec(
            CURRENT_MODEL, "Boosted trees",
            "The pipeline's production model: CatBoost with model.py's settings, learning the annual value (bookings ÷ term years) and multiplying back by the term.",
            lambda: _CatBoost(annualized_target=True), True,
        ))
        specs.append(ModelSpec(
            "CatBoost (total-bookings target)", "Boosted trees",
            "The previous production setup: same CatBoost settings, but learning total bookings directly.",
            lambda: _CatBoost(annualized_target=False), True,
        ))
    return specs


# ==========================================================================
# EVALUATION HELPERS (the checks beyond plain accuracy)
# ==========================================================================
def _clip(values) -> np.ndarray:
    """Bookings cannot be negative, so any negative prediction is set to 0."""
    return np.clip(np.asarray(values, dtype=float), 0, None)


def _with_term(frame: pd.DataFrame, term: str) -> pd.DataFrame:
    """Copy of the deals with TERM_BUCKET overwritten, e.g. 'what if every deal were 5 Year?'.

    Everything else about each deal (customer, suite, segment...) stays the
    same. This is the same counterfactual trick the what-if engine uses.
    """
    scenario = frame.copy()
    scenario["TERM_BUCKET"] = term
    return scenario


def scenario_check(model, test: pd.DataFrame, train: pd.DataFrame) -> dict[str, float]:
    """Does the model react sensibly when ONLY the term changes?

    Each test deal is scored three times (as 1, 3 and 5 years). We then check:
    - monotonic_pct: % of deals where 1-year < 3-year < 5-year bookings
      (a longer contract should commit more money in total)
    - annual_ratio_*: the model's implied annual value of a 1-/5-year deal
      relative to a 3-year deal (1.00 = same annual value)
    - default_scenario_change_pct: the what-if answer this model would give
      at 100% flexible adoption and 50% five-year share
    """
    # p1, p3, p5 = predicted bookings for every test deal as a 1/3/5-year contract.
    p1, p3, p5 = (_clip(model.predict(_with_term(test, term))) for term in TERMS)
    safe3 = np.where(p3 > 0, p3, np.nan)   # avoid dividing by zero below
    annual3 = safe3 / 3                    # annual value of the 3-year version
    # Raw medians from history, for comparison (these are confounded: big customers pick 5-year more).
    observed = (train[TARGET] / _term_years(train["TERM_BUCKET"])).groupby(train["TERM_BUCKET"].astype(str)).median()
    # Default app scenario: 100% flexible adoption, 50% five-year share,
    # i.e. every deal becomes half 1-year and half 5-year.
    mixed = 0.5 * p1 + 0.5 * p5
    return {
        "monotonic_pct": float(np.mean((p1 < p3) & (p3 < p5)) * 100),
        # % of deals where switching term barely changes the prediction (model ignoring the lever).
        "flat_term_pct": float(np.mean(np.isclose(p1, p3, rtol=0.01) | np.isclose(p3, p5, rtol=0.01)) * 100),
        "annual_ratio_1y_vs_3y": float(np.nanmedian(p1 / annual3)),
        "annual_ratio_5y_vs_3y": float(np.nanmedian((p5 / 5) / annual3)),
        "observed_ratio_1y_vs_3y": float(observed.get("1 Year", np.nan) / observed.get("3 Year", np.nan)),
        "observed_ratio_5y_vs_3y": float(observed.get("5 Year", np.nan) / observed.get("3 Year", np.nan)),
        # Total mixed-scenario bookings vs all-3-year bookings, in %.
        "default_scenario_change_pct": float((mixed.sum() - p3.sum()) / p3.sum() * 100) if p3.sum() else np.nan,
    }


# Slider defaults on the main app page (Deals using flexible terms,
# Five-year share of flexible deals, Annual discount rate).
DEFAULT_ADOPTION_PCT, DEFAULT_FIVE_YEAR_SHARE_PCT, DEFAULT_DISCOUNT_RATE_PCT = 100, 50, 8


def _lens_outputs(term_bookings: dict[int, np.ndarray], frame: pd.DataFrame) -> dict[str, float]:
    """Run the app's own scenario.py maths and return the three-lens numbers.

    term_bookings = {1: bookings if 1-year, 3: ..., 5: ...} for each deal.
    Using scenario.py (not a copy of its formulas) guarantees these numbers
    are exactly what the main page would show if it used this model.
    """
    from .scenario import annual_coverage, breakeven_share, calculate_scenario, sensitivity, summarise

    # Build the same columns predict_term_options() produces for the app.
    scored = frame[["SUBSCRIPTION_REFERENCE_ID"]].reset_index(drop=True).copy()
    for years, values in term_bookings.items():
        scored[f"bookings_{years}y"] = values
        # Lens check uses point estimates only; ranges come from the quantile models.
        scored[f"bookings_{years}y_p10"] = values
        scored[f"bookings_{years}y_p90"] = values
    details = calculate_scenario(scored, DEFAULT_ADOPTION_PCT, DEFAULT_FIVE_YEAR_SHARE_PCT, DEFAULT_DISCOUNT_RATE_PCT)
    coverage = annual_coverage(details, DEFAULT_ADOPTION_PCT, DEFAULT_FIVE_YEAR_SHARE_PCT)
    summary = summarise(details, coverage)
    # Breakeven = lowest five-year share at which NPV stops falling below the 3-year baseline.
    breakeven = breakeven_share(sensitivity(scored, DEFAULT_ADOPTION_PCT, DEFAULT_DISCOUNT_RATE_PCT))
    return {
        # Lens 1 · Growth: do bookings go up, and is the money worth it after discounting?
        "growth_commitment_change_pct": summary["commitment_change_pct"],
        "growth_npv_change_pct": summary["npv_change"] / summary["baseline_npv"] * 100 if summary["baseline_npv"] else np.nan,
        "growth_npv_breakeven_5y_share_pct": np.nan if breakeven is None else breakeven,
        # Lens 2 · Customer value (lever-driven proxy: identical for every model,
        # because the app computes it from the sliders, not from predictions)
        "customer_avg_term_years": summary["average_term_years"],
        # Lens 3 · Simplicity with scale: how bumpy is contracted value year to
        # year, and how many extra renewal decisions does the mix create?
        "simplicity_coverage_cv_pct": summary["scenario_coverage_cv_pct"],
        "simplicity_coverage_cv_change_pts": summary["scenario_coverage_cv_pct"] - summary["baseline_coverage_cv_pct"],
        "simplicity_renewal_points_change": summary["scenario_renewal_points"] - summary["baseline_renewal_points"],
    }


def lens_check(model, test: pd.DataFrame) -> dict[str, float]:
    """Three business lenses (PPT slide 3) computed from this model's 1/3/5-year scores."""
    return _lens_outputs({TERM_YEARS[t]: _clip(model.predict(_with_term(test, t))) for t in TERMS}, test)


def synthetic_truth_lenses(test: pd.DataFrame) -> dict[str, float]:
    """Exact lens values on synthetic data, using the generator's known term factors.

    For each deal we recover its underlying annual value (divide out the
    factor of the term it actually had), then rebuild what it would have
    booked as 1, 3 or 5 years. This is the 'answer key' models are graded against.
    """
    own_factor = _term_years(test["TERM_BUCKET"]).map(SYNTHETIC_TERM_ANNUAL_FACTOR).to_numpy(float)
    base_annual = test["ANNUALIZED_NET_PRICE_USD_AMT"].to_numpy(float) / own_factor
    return _lens_outputs({t: base_annual * SYNTHETIC_TERM_ANNUAL_FACTOR[t] * t for t in (1, 3, 5)}, test)


def backtest_folds(train_validation: pd.DataFrame, n_folds: int = 3) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Expanding-window folds over distinct start dates (never touches test).

    Example with 3 folds over the train + validation period:
      fold 1: learn from the first 50% of dates  -> score the next ~17%
      fold 2: learn from the first ~67% of dates -> score the next ~17%
      fold 3: learn from the first ~83% of dates -> score the last ~17%
    Always "learn from the past, predict the future", like real use.
    The spread of WAPE across folds shows how stable a model is.
    """
    data = train_validation.dropna(subset=["TERM_START_DATE"]).sort_values("TERM_START_DATE")
    dates = np.array(sorted(data["TERM_START_DATE"].unique()))
    if len(dates) < n_folds + 2:
        return []   # too little history to backtest
    edges = np.linspace(0.5, 1.0, n_folds + 1)   # e.g. [0.50, 0.67, 0.83, 1.00]
    folds = []
    for start, end in zip(edges[:-1], edges[1:]):
        cut = pd.Timestamp(dates[int(len(dates) * start) - 1])                     # last date used for learning
        stop = pd.Timestamp(dates[min(int(len(dates) * end), len(dates)) - 1])     # last date scored
        fit_part = data[data.TERM_START_DATE <= cut]
        eval_part = data[(data.TERM_START_DATE > cut) & (data.TERM_START_DATE <= stop)]
        if len(fit_part) and len(eval_part):
            folds.append((fit_part, eval_part))
    return folds


@dataclass
class ComparisonResult:
    """Everything different_models() produces; each table is saved as one CSV."""
    leaderboard: pd.DataFrame       # one row per model: all metrics, time, rank (leaderboard.csv)
    metrics_by_term: pd.DataFrame   # metrics split by 1/3/5-year deals (metrics_by_term.csv)
    backtest: pd.DataFrame          # one row per model per fold (backtest.csv)
    scenario: pd.DataFrame          # scenario fitness per model (scenario_check.csv)
    predictions: pd.DataFrame       # every test deal's actual vs predicted, per model (test_predictions.csv)
    metadata: dict[str, Any] = field(default_factory=dict)        # run info + recommendation (metadata.json)
    lenses: pd.DataFrame = field(default_factory=pd.DataFrame)    # three business lenses (business_lenses.csv)


def _recommend(board: pd.DataFrame) -> tuple[str, str]:
    """Pick the model to recommend and explain why, in plain English.

    Rules:
    1. Only consider scenario-safe models (they keep 1y < 3y < 5y), excluding the baseline.
    2. The best = lowest test WAPE (ties broken by the more stable model).
    3. Only recommend switching away from the current CatBoost if the best
       model wins by MORE than CatBoost's own fold-to-fold spread. A smaller
       gap is within normal noise and is not worth a production change.
    """
    safe = board[board.scenario_safe & (board.model != BASELINE_MODEL)]
    pool = safe if not safe.empty else board[board.model != BASELINE_MODEL]
    best = pool.sort_values(["test_wape_pct", "backtest_wape_std"]).iloc[0]
    current = board[board.model == CURRENT_MODEL]
    if current.empty:
        return best.model, "CatBoost not installed; best scenario-safe model by test WAPE."
    cur = current.iloc[0]
    gap = cur.test_wape_pct - best.test_wape_pct   # how many WAPE points the best model wins by
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


# ==========================================================================
# MAIN FUNCTION
# ==========================================================================
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
        Prepared features (``data/prepared/term_model_features.parquet``).
    models:
        Optional subset of model names from ``model_catalogue()``; default all.
        Example: different_models(frame, models=["CatBoost (current)", "LightGBM"])
    n_backtest_folds:
        Expanding-window folds for the stability check; 0 disables it (faster).
    progress:
        Optional callback receiving status text (used by the Streamlit page).

    Steps, for EACH model:
      A. Train on the train period (validation only for early stopping); time it.
      B. Score validation and test periods -> accuracy metrics, overall and per term.
      C. Scenario check  -> does it react sensibly to 1/3/5-year terms?
      D. Lens check      -> Growth / Customer value / Simplicity numbers.
      E. Backtest        -> retrain on earlier periods to measure stability.
      F. Add one row to the leaderboard.
    Then, once all models are done:
      G. Rank by test WAPE, compare with the baseline and with the known truth
         (synthetic data only), and pick a recommendation.
    """
    # Where status messages go: the Streamlit page passes its own; otherwise the log.
    say = progress or (lambda message: LOGGER.info(message))
    specs = [s for s in model_catalogue() if models is None or s.name in models]
    if not specs:
        raise ValueError("No models selected")

    # Same split as production: train = oldest dates, validation = middle, test = newest.
    # The test period is never used for training or tuning, so it is a fair exam.
    splits = chronological_split(frame)
    train, validation, test = splits["train"], splits["validation"], splits["test"]
    train_validation = pd.concat([train, validation])
    folds = backtest_folds(train_validation, n_backtest_folds) if n_backtest_folds else []

    # Each list collects one result table; every model appends its rows.
    board_rows, term_rows, cv_rows, scenario_rows, lens_rows, prediction_frames = [], [], [], [], [], []
    for spec in specs:
        # ---- A. Train and time it -------------------------------------------
        say(f"Training {spec.name}")
        model = spec.builder()              # fresh, untrained model
        started = time.perf_counter()
        model.fit(train, validation if spec.uses_validation else None)
        fit_seconds = time.perf_counter() - started

        # ---- B. Accuracy on validation and test ------------------------------
        split_metrics = {}
        for split_name, part in (("validation", validation), ("test", test)):
            started = time.perf_counter()
            predicted = _clip(model.predict(part))
            predict_ms = (time.perf_counter() - started) / max(len(part), 1) * 1000   # ms per deal
            actual = part[TARGET].to_numpy(float)
            split_metrics[split_name] = regression_metrics(actual, predicted)
            # Same metrics for 1-, 3- and 5-year deals separately (is it good for all terms?).
            for term in TERMS:
                mask = part["TERM_BUCKET"].eq(term).to_numpy()
                term_rows.append({"model": spec.name, "split": split_name, "term_bucket": term, "rows": int(mask.sum()), **regression_metrics(actual[mask], predicted[mask])})
            if split_name == "test":
                # Keep every test prediction for the dashboard's scatter plot and histogram.
                prediction_frames.append(pd.DataFrame({
                    "model": spec.name, "SUBSCRIPTION_REFERENCE_ID": part["SUBSCRIPTION_REFERENCE_ID"].to_numpy(),
                    "TERM_START_DATE": part["TERM_START_DATE"].to_numpy(), "TERM_BUCKET": part["TERM_BUCKET"].to_numpy(),
                    "SUITE_NAME": part["SUITE_NAME"].to_numpy(), "SCMS": part["SCMS"].to_numpy(),
                    "actual": actual, "predicted": predicted,
                }))

        # ---- C. Scenario check and D. three business lenses ------------------
        check = scenario_check(model, test, train)
        if is_synthetic_dataset():
            # On synthetic data we know the right answer, so record how far off this model is.
            check["synthetic_true_change_pct"] = SYNTHETIC_TRUE_DEFAULT_CHANGE_PCT
            check["scenario_error_vs_truth_pts"] = check["default_scenario_change_pct"] - SYNTHETIC_TRUE_DEFAULT_CHANGE_PCT
        scenario_rows.append({"model": spec.name, **check})
        lens_rows.append({"model": spec.name, **lens_check(model, test)})

        # ---- E. Backtest: retrain on earlier periods, score the next period ---
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

        # ---- F. One leaderboard row for this model ---------------------------
        test_m, val_m = split_metrics["test"], split_metrics["validation"]
        board_rows.append({
            "model": spec.name, "family": spec.family, "description": spec.description,
            **{f"test_{k}": v for k, v in test_m.items()},       # test_mae, test_wape_pct, ...
            "validation_wape_pct": val_m["wape_pct"],
            "backtest_wape_mean": float(np.mean(fold_wapes)) if fold_wapes else np.nan,   # typical error over time
            "backtest_wape_std": float(np.std(fold_wapes)) if fold_wapes else np.nan,     # how much it moves = noise
            "fit_seconds": fit_seconds, "predict_ms_per_row": predict_ms,
            "monotonic_pct": check["monotonic_pct"],
            "default_scenario_change_pct": check["default_scenario_change_pct"],
            "scenario_safe": check["monotonic_pct"] >= MONOTONIC_THRESHOLD_PCT,
        })
        LOGGER.info("%s: test WAPE %.2f%%, monotonic %.1f%%", spec.name, test_m["wape_pct"], check["monotonic_pct"])

    # ---- G. Rank, compare, recommend ----------------------------------------
    # Rank 1 = lowest test WAPE.
    board = pd.DataFrame(board_rows).sort_values("test_wape_pct", ignore_index=True)
    board.insert(0, "rank", range(1, len(board) + 1))
    # How much better than "no ML" each model is, in % of the baseline's error.
    baseline = board.loc[board.model == BASELINE_MODEL, "test_wape_pct"]
    board["wape_gain_vs_baseline_pct"] = (
        (baseline.iloc[0] - board.test_wape_pct) / baseline.iloc[0] * 100 if not baseline.empty else np.nan
    )
    # Business lenses table, plus a "known truth" row and per-model errors on synthetic data.
    lenses = pd.DataFrame(lens_rows)
    if is_synthetic_dataset():
        truth = synthetic_truth_lenses(test)
        for column in ("growth_commitment_change_pct", "growth_npv_change_pct", "simplicity_coverage_cv_pct"):
            lenses[column.replace("_pct", "") + "_error_vs_truth_pts"] = lenses[column] - truth[column]
        lenses = pd.concat([lenses, pd.DataFrame([{"model": SYNTHETIC_TRUTH_LABEL, **truth}])], ignore_index=True)
    recommended, reason = _recommend(board)
    board["recommended"] = board.model.eq(recommended)

    # Run information saved with the results, for traceability (shown at the top of the dashboard).
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


# ==========================================================================
# SAVE / LOAD
# ==========================================================================
def save_comparison(result: ComparisonResult, output_dir: Path = COMPARISON_DIR) -> dict[str, Path]:
    """Write every result table to data/models/model_comparison/ (read by the dashboard)."""
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
    """Read saved results back, so the dashboard opens instantly without retraining."""
    read = lambda name: pd.read_csv(output_dir / name)
    return ComparisonResult(
        read("leaderboard.csv"), read("metrics_by_term.csv"),
        # backtest.csv is empty when folds = 0, so only read it if it has content.
        read("backtest.csv") if (output_dir / "backtest.csv").stat().st_size > 2 else pd.DataFrame(),
        read("scenario_check.csv"), read("test_predictions.csv"),
        json.loads((output_dir / "metadata.json").read_text(encoding="utf-8")),
        # business_lenses.csv only exists for runs made after the lens check was added.
        read("business_lenses.csv") if (output_dir / "business_lenses.csv").exists() else pd.DataFrame(),
    )
