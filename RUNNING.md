# How to run this project

Quick-start guide for the Cluster C ML Scenario (flexible 1- or 5-year terms vs 3-year).
For background, see [README.md](README.md). For data rules, see [DATA_CONTRACT.md](DATA_CONTRACT.md).

## 1. One-time setup

Run these from the project folder (the one that contains `app.py`).

**Windows (Command Prompt)**

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. Every new terminal

Activate the environment and tell Python where the code lives (`src`).

| Terminal | Commands |
|---|---|
| Windows Command Prompt | `.venv\Scripts\activate` then `set PYTHONPATH=src` |
| Windows PowerShell | `.venv\Scripts\Activate.ps1` then `$env:PYTHONPATH="src"` |
| macOS / Linux | `source .venv/bin/activate` then `export PYTHONPATH=src` |

> Tip: if the prompt starts with `C:\...>` you are in Command Prompt; if it starts with `PS C:\...>` you are in PowerShell.

## 3. Run the pipeline

Run in this order. Steps 3a and 3b are only needed when the input data changes,
because the repository already includes prepared data and trained models.

| Step | Command (Windows) | What it does | Output |
|---|---|---|---|
| 3a. Prepare data | `python scripts\prepare_data.py` | Checks `data\input\term_deals.csv` against the data contract and builds features | `data\prepared\` |
| 3b. Train CatBoost | `python scripts\train_model.py` | Trains the production models (main estimate, P10, P90) | `data\models\cluster_c_term_v1\` |
| 3c. Compare models | `python scripts\compare_models.py` | Trains 9 models and compares accuracy, stability and the three business lenses (about 1–2 min) | `data\models\model_comparison\` |
| 3d. Open the app | `streamlit run app.py` | Opens the browser. Main page = the what-if; sidebar **Model Comparison** = model results | — |

### Compare models on a larger synthetic dataset

```bat
python scripts\compare_models.py --rows 50000
python scripts\compare_models.py --rows 100000
python scripts\compare_models.py --rows 100000 --folds 0
```

`--rows N` generates N synthetic deals in memory with the same generator and seed as the sample, prepares them, compares the 9 models and saves the results to `data\models\model_comparison_N_rows\`. Your sample (`data\input\term_deals.csv`) and its prepared files are not changed. In the app, choose the dataset under **Result set** in the Model Comparison sidebar.

| Option | Meaning |
|---|---|
| `--rows N` | Number of synthetic deals to generate (at least 100) |
| `--folds K` | Backtest folds for the stability check (default 3; `0` skips it and is about 4× faster) |
| `--seed S` | Random seed for the generated data (default 42) |
| `--save-data PATH` | Also save the generated raw CSV, e.g. `--save-data data\input\term_deals_100k.csv` (cannot overwrite `term_deals.csv`) |

Approximate run time (9 models, 3 folds): 50,000 rows about 3 minutes, 100,000 rows about 6 minutes.

On macOS/Linux use `/` instead of `\` in the paths, e.g. `python scripts/prepare_data.py`.

Optional: regenerate the synthetic sample data first with `python scripts\generate_sample_data.py`.

Using new Cisco data: save the file as `data\input\term_deals.csv`, set a version label
(`set SOURCE_DATASET_VERSION=cisco-cluster-c-YYYYMMDD-v1` in Command Prompt), then run 3a → 3d.

## 4. Run the tests (pytest)

```bat
pytest -q
```

Expected result: **16 passed** (takes under a minute). Useful variations:

| Command | Use |
|---|---|
| `pytest -q` | Run all tests, short output |
| `pytest -v` | Run all tests, one line per test |
| `pytest -q tests\test_model_comparison.py` | Run only one test file |
| `pytest -q -k lens` | Run only tests whose name contains "lens" |

### What the tests check

| Test file | Test | What it proves |
|---|---|---|
| `test_pipeline.py` | `test_synthetic_data_is_deterministic_and_matches_contract` | Sample data is identical every run and has all required columns |
| | `test_preparation_passes_and_prior_history_does_not_use_current_target` | Customer-history features never peek at the deal's own bookings (no leakage) |
| | `test_row_level_contract_failures_are_non_blocking` | Bad rows are reported in `quality_checks.csv` but do not stop preparation |
| | `test_common_term_bucket_aliases_are_normalized` | Labels like `12 Months`, `36_months`, `5 years` become `1 Year` / `3 Year` / `5 Year` |
| `test_model.py` | `test_saved_model_scores_all_three_terms` | Saved models score every deal as 1/3/5 years with P10 ≤ estimate ≤ P90; split is in date order |
| | `test_small_diagnostic_dataset_uses_chronological_fallback` | Very small datasets still split into train/validation/test without overlap |
| | `test_metrics_are_nan_for_an_empty_term_slice` | Metrics return "not available" instead of crashing when a term has no rows |
| `test_scenario.py` | `test_flexible_mix_and_coverage_are_transparent` | What-if maths gives hand-checkable answers (commitment, average term, renewal points) |
| | `test_sensitivity_and_breakeven` | Sensitivity curve has 21 points (0–100% in 5% steps) and finds a breakeven |
| `test_app.py` | `test_single_page_streamlit_ui` | Main app page loads without errors and the sliders work |
| `test_model_comparison.py` | `test_regression_metrics_on_known_values` | MAE / WAPE / bias formulas give the right numbers on a small worked example |
| | `test_different_models_fast_subset` | `different_models()` runs end to end and every model is scored on the same test deals |
| | `test_backtest_folds_never_look_ahead` | Backtest folds always learn from the past and score the future |
| | `test_catalogue_includes_current_catboost` | The production CatBoost is part of the comparison |
| | `test_model_comparison_page_renders` | The Model Comparison page loads without errors |
| | `test_lens_check_matches_known_synthetic_truth_inputs` | The "known truth" used to grade what-if answers equals +5.25% |

## 5. Common problems

| Message | Fix |
|---|---|
| `The filename, directory name, or volume label syntax is incorrect` after `$env:...` | You are in Command Prompt; use `set PYTHONPATH=src` instead |
| `ModuleNotFoundError: No module named 'cluster_c'` | Run the `PYTHONPATH` command from step 2 in this terminal |
| `ModuleNotFoundError: No module named 'catboost'` (or xgboost, lightgbm, sklearn) | Activate `.venv`, then `python -m pip install -r requirements.txt` |
| `running scripts is disabled on this system` (PowerShell) | Run `Set-ExecutionPolicy -Scope Process Bypass`, then activate again |
| Model Comparison page says "No comparison results yet" | Run `python scripts\compare_models.py` or click **Run comparison now** in the page sidebar |
