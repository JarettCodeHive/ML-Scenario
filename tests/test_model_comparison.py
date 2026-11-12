from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from cluster_c.config import PREPARED_FEATURE_PATH
from cluster_c.model_comparison import (
    BASELINE_MODEL,
    CURRENT_MODEL,
    backtest_folds,
    different_models,
    model_catalogue,
    regression_metrics,
)


def _frame() -> pd.DataFrame:
    return pd.read_parquet(PREPARED_FEATURE_PATH, engine="pyarrow")


def test_regression_metrics_on_known_values() -> None:
    result = regression_metrics(np.array([100.0, 200.0]), np.array([110.0, 180.0]))
    assert result["mae"] == 15.0
    assert round(result["wape_pct"], 6) == 10.0
    assert round(result["bias_pct"], 6) == round(-10 / 300 * 100, 6)
    assert result["within_20pct"] == 100.0
    assert all(np.isnan(v) for v in regression_metrics([], []).values())


def test_different_models_fast_subset() -> None:
    frame = _frame()
    result = different_models(frame, models=[BASELINE_MODEL, "Ridge regression (log target)"], n_backtest_folds=2)
    board = result.leaderboard
    assert set(board.model) == {BASELINE_MODEL, "Ridge regression (log target)"}
    assert board.test_wape_pct.is_monotonic_increasing
    assert (result.backtest.groupby("model").size() == 2).all()
    assert set(result.scenario.columns) >= {"monotonic_pct", "default_scenario_change_pct"}
    assert result.metadata["recommended_model"] in board.model.tolist()
    # Same untouched test rows for every model.
    assert result.predictions.groupby("model").size().nunique() == 1


def test_backtest_folds_never_look_ahead() -> None:
    frame = _frame()
    for fit_part, eval_part in backtest_folds(frame, 3):
        assert fit_part.TERM_START_DATE.max() < eval_part.TERM_START_DATE.min()


def test_catalogue_includes_current_catboost() -> None:
    assert CURRENT_MODEL in [spec.name for spec in model_catalogue()]


def test_model_comparison_page_renders() -> None:
    page = Path(__file__).resolve().parents[1] / "pages" / "1_Model_Comparison.py"
    app = AppTest.from_file(str(page), default_timeout=60).run()
    assert not app.exception


def test_lens_check_matches_known_synthetic_truth_inputs() -> None:
    from cluster_c.model_comparison import synthetic_truth_lenses

    frame = _frame()
    truth = synthetic_truth_lenses(frame.head(200))
    assert round(truth["growth_commitment_change_pct"], 2) == 5.25
    assert truth["customer_avg_term_years"] == 3.0


def test_compare_models_rows_option_writes_its_own_result_folder(tmp_path, monkeypatch) -> None:
    """--rows N generates data in memory and saves to model_comparison_N_rows, not the sample folder."""
    import sys

    import scripts.compare_models as cli

    monkeypatch.setattr(cli, "DATA_DIR", tmp_path)
    monkeypatch.setattr(cli, "COMPARISON_DIR", tmp_path / "models" / "model_comparison")
    monkeypatch.setattr(cli, "different_models", lambda frame, n_backtest_folds=3: different_models(
        frame, models=[BASELINE_MODEL, "Ridge regression (log target)"], n_backtest_folds=0))
    monkeypatch.setattr(sys, "argv", ["compare_models.py", "--rows", "1200", "--folds", "0"])
    cli.main()

    folder = tmp_path / "models" / "model_comparison_1200_rows"
    assert (folder / "leaderboard.csv").exists()
    assert not (tmp_path / "models" / "model_comparison").exists()
    import json
    assert json.loads((folder / "metadata.json").read_text(encoding="utf-8"))["source_rows"] == 1200
