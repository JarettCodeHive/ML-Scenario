import numpy as np
import pandas as pd
import pytest

from cluster_c.model import (
    TRAINING_COLUMNS,
    TRAINING_SAMPLE_STRATA,
    TrainingConfig,
    _new_model,
    chronological_split,
    matrix,
    metrics,
    predict_term_options,
    representative_training_sample,
    save_result,
    training_eligible_frame,
    train_models,
)
from cluster_c.preparation import (
    CATEGORICAL_FEATURES,
    EXCLUDED_IDENTIFIER_FEATURES,
    FEATURES,
    prepare,
)
from cluster_c.synthetic import generate_term_deals
from scripts.train_model import value_counts_for_log


def test_multi_quantile_model_scores_all_three_terms(tmp_path) -> None:
    frame = prepare(generate_term_deals(row_count=500)).frame
    splits = chronological_split(frame)
    assert splits["train"].TERM_START_DATE.max() < splits["validation"].TERM_START_DATE.min()
    assert splits["validation"].TERM_START_DATE.max() < splits["test"].TERM_START_DATE.min()

    result = train_models(
        frame,
        config=TrainingConfig(iterations=3, depth=3, early_stopping_rounds=3),
    )
    paths = save_result(result, tmp_path)
    predictions = predict_term_options(frame.head(5), tmp_path)

    assert paths["interval"].name == "bookings_interval_10_90.cbm"
    assert "lower" not in paths
    assert "upper" not in paths
    assert result.metadata["model_count"] == 2
    assert result.metadata["interval_loss"] == "MultiQuantile:alpha=0.1,0.9"
    assert len(predictions) == 5
    for years in (1, 3, 5):
        assert (predictions[f"bookings_{years}y"] >= 0).all()
        assert (predictions[f"bookings_{years}y_p10"] <= predictions[f"bookings_{years}y"]).all()
        assert (predictions[f"bookings_{years}y"] <= predictions[f"bookings_{years}y_p90"]).all()


def test_small_diagnostic_dataset_uses_chronological_fallback() -> None:
    frame = pd.DataFrame({
        "TERM_START_DATE": pd.date_range("2024-01-01", periods=8, freq="D").strftime("%Y-%m-%d"),
        "TERM_BUCKET": ["3 Year", "5 Year", "3 Year", "3 Year", "3 Year", "1 Year", "3 Year", "1 Year"],
    })
    original = frame.copy()

    splits = chronological_split(frame)

    pd.testing.assert_frame_equal(frame, original)
    assert all(pd.api.types.is_datetime64_any_dtype(part["TERM_START_DATE"]) for part in splits.values())
    assert all(not part.empty for part in splits.values())
    assert splits["train"].TERM_START_DATE.max() < splits["validation"].TERM_START_DATE.min()
    assert splits["validation"].TERM_START_DATE.max() < splits["test"].TERM_START_DATE.min()


def test_metrics_are_nan_for_an_empty_term_slice() -> None:
    result = metrics(np.array([]), np.array([]))

    assert all(np.isnan(value) for value in result.values())


def test_thread_count_is_configurable_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("CATBOOST_THREAD_COUNT", "4")

    assert TrainingConfig().thread_count == 4


def test_training_log_counts_null_categorical_values_without_adding_category() -> None:
    terms = pd.Series(
        pd.Categorical(
            ["1 Year", None, "3 Year", None, "1 Year"],
            categories=["1 Year", "3 Year", "5 Year"],
        )
    )

    counts = value_counts_for_log(terms)

    assert counts == {"1 Year": 2, "<NULL>": 2, "3 Year": 1}
    assert "<NULL>" not in terms.cat.categories


def test_training_excludes_only_rows_without_a_finite_target() -> None:
    frame = pd.DataFrame({
        "BOOKINGS_NET_PRICE_USD_AMT": [100.0, np.nan, np.inf, -25.0],
        "row_id": [1, 2, 3, 4],
    })

    eligible = training_eligible_frame(frame)

    assert eligible["row_id"].tolist() == [1, 4]


def test_invalid_thread_count_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("CATBOOST_THREAD_COUNT", "0")

    with pytest.raises(ValueError, match="CATBOOST_THREAD_COUNT"):
        TrainingConfig()


def test_high_cardinality_identifiers_are_not_model_features() -> None:
    excluded = set(EXCLUDED_IDENTIFIER_FEATURES)
    prepared = prepare(generate_term_deals(row_count=30)).frame
    values = matrix(prepared)

    assert excluded.isdisjoint(FEATURES)
    assert excluded.isdisjoint(CATEGORICAL_FEATURES)
    assert excluded.isdisjoint(values.columns)
    assert list(values.columns) == FEATURES
    assert "CAV_END_CUSTOMER_BU_ID" in prepared.columns
    assert "UCD_C4_ACCOUNT_ID" in prepared.columns
    assert "CAV_END_CUSTOMER_BU_ID" in TRAINING_COLUMNS
    assert "UCD_C4_ACCOUNT_ID" not in TRAINING_COLUMNS
    assert all(values[column].dtype == "float32" for column in set(FEATURES) - set(CATEGORICAL_FEATURES))


def test_categorical_complexity_controls_are_passed_to_catboost() -> None:
    config = TrainingConfig(max_ctr_complexity=1, ctr_leaf_count_limit=100_000)
    params = _new_model(config, "MAE").get_params()

    assert params["max_ctr_complexity"] == 1
    assert params["ctr_leaf_count_limit"] == 100_000


@pytest.mark.parametrize(
    ("setting", "value"),
    [("max_ctr_complexity", 0), ("ctr_leaf_count_limit", 0)],
)
def test_invalid_categorical_complexity_controls_are_rejected(setting, value) -> None:
    with pytest.raises(ValueError, match=setting):
        TrainingConfig(**{setting: value})


def test_optimized_boosting_defaults_are_passed_to_catboost() -> None:
    config = TrainingConfig()
    params = _new_model(config, "MAE").get_params()

    assert config.iterations == 400
    assert config.depth == 6
    assert config.learning_rate == 0.06
    assert config.early_stopping_rounds == 40
    assert params["iterations"] == 400
    assert params["depth"] == 6
    assert params["learning_rate"] == 0.06


@pytest.mark.parametrize(
    ("setting", "value"),
    [("iterations", 0), ("depth", 0), ("learning_rate", 0), ("early_stopping_rounds", 0)],
)
def test_invalid_boosting_configuration_is_rejected(setting, value) -> None:
    with pytest.raises(ValueError, match=setting):
        TrainingConfig(**{setting: value})


def test_representative_training_sample_is_exact_and_deterministic() -> None:
    frame = pd.DataFrame({
        "FISCAL_YEAR": np.repeat([2024, 2025], 100),
        "TERM_BUCKET": np.tile(np.repeat(["1 Year", "3 Year"], 50), 2),
        "SCMS": np.tile(["Enterprise", "Commercial"], 100),
        "SUITE_NAME": np.tile(["Security", "Networking"], 100),
        "row_id": np.arange(200),
    })

    first = representative_training_sample(frame, max_rows=40, random_seed=42)
    second = representative_training_sample(frame, max_rows=40, random_seed=42)

    assert len(first) == 40
    assert first.index.tolist() == second.index.tolist()
    assert first.groupby(TRAINING_SAMPLE_STRATA, observed=True).ngroups == frame.groupby(
        TRAINING_SAMPLE_STRATA, observed=True
    ).ngroups


def test_training_sample_size_is_configurable_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("TRAINING_SAMPLE_ROWS", "250000")
    assert TrainingConfig().max_training_rows == 250_000

    monkeypatch.setenv("TRAINING_SAMPLE_ROWS", "all")
    assert TrainingConfig().max_training_rows is None


def test_training_uses_all_rows_by_default(monkeypatch) -> None:
    monkeypatch.delenv("TRAINING_SAMPLE_ROWS", raising=False)

    assert TrainingConfig().max_training_rows is None


def test_training_metadata_records_sampling() -> None:
    frame = prepare(generate_term_deals(row_count=500)).frame
    result = train_models(
        frame,
        config=TrainingConfig(
            iterations=2,
            depth=3,
            early_stopping_rounds=2,
            max_training_rows=100,
        ),
    )

    sampling = result.metadata["training_sampling"]
    assert sampling["enabled"] is True
    assert sampling["source_training_rows"] > sampling["sampled_training_rows"]
    assert sampling["sampled_training_rows"] == 100
    assert sampling["strata"] == TRAINING_SAMPLE_STRATA
