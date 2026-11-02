import numpy as np
import pandas as pd

from cluster_c.config import MODEL_DIR, PREPARED_DIR
from cluster_c.model import chronological_split, metrics, predict_term_options


def test_saved_model_scores_all_three_terms() -> None:
    frame = pd.read_csv(PREPARED_DIR / "term_model_features.csv", parse_dates=["TERM_START_DATE", "TERM_END_DATE"])
    splits = chronological_split(frame)
    assert splits["train"].TERM_START_DATE.max() < splits["validation"].TERM_START_DATE.min()
    assert splits["validation"].TERM_START_DATE.max() < splits["test"].TERM_START_DATE.min()
    predictions = predict_term_options(frame.head(5), MODEL_DIR)
    assert len(predictions) == 5
    for years in (1, 3, 5):
        assert (predictions[f"bookings_{years}y"] >= 0).all()
        assert (predictions[f"bookings_{years}y_p10"] <= predictions[f"bookings_{years}y"]).all()
        assert (predictions[f"bookings_{years}y"] <= predictions[f"bookings_{years}y_p90"]).all()


def test_small_diagnostic_dataset_uses_chronological_fallback() -> None:
    frame = pd.DataFrame({
        "TERM_START_DATE": pd.date_range("2024-01-01", periods=8, freq="D"),
        "TERM_BUCKET": ["3 Year", "5 Year", "3 Year", "3 Year", "3 Year", "1 Year", "3 Year", "1 Year"],
    })

    splits = chronological_split(frame)

    assert all(not part.empty for part in splits.values())
    assert splits["train"].TERM_START_DATE.max() < splits["validation"].TERM_START_DATE.min()
    assert splits["validation"].TERM_START_DATE.max() < splits["test"].TERM_START_DATE.min()


def test_metrics_are_nan_for_an_empty_term_slice() -> None:
    result = metrics(np.array([]), np.array([]))

    assert all(np.isnan(value) for value in result.values())
