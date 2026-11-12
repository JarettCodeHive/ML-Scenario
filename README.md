#  EA Cluster C ML Scenario

A standalone local proof of concept aligned to the EA Lever Modelling Scope v4:

> **Cluster:** C — Term and Renewal  
> **What-if scenario:** Flexible 1- or 5-year terms against 3-year  
> **Core question:** Does term flexibility grow commitments or destabilise revenue?  
> **PPT register:** Wave 1 · Scenario 8 · Type 1

The project contains only this scenario: synthetic -shaped data, quality checks, point-in-time feature preparation, local CatBoost training, deterministic portfolio calculations, and a one-page Streamlit decision view.

> **Quick start:** see [RUNNING.md](RUNNING.md) for step-by-step run commands (Windows and macOS) and what every pytest checks.

## PPT alignment

The UI follows the PPT decision structure:

- One business question and one lever cluster
- 3-year signed reference against a configurable 1-/5-year mix
- Growth, customer value, and simplicity-with-scale lenses
- Total commitment and NPV
- Experimental P10/P50/P90 estimates
- Breakeven and sensitivity to five-year share
- Visible assumptions, limitations, model metrics, and calculation flow
- Filters for sales level, segment, and suite
- Contract/deal-level export

The supplied CSV does not contain renewal, churn, acceptance, invoice, or operating-cost outcomes. The corresponding UI measures are explicitly labelled proxies rather than claims.

## Technology

- Python
- Pandas and NumPy
- CatBoost
- pypdf
- Streamlit
- Plotly
- pytest

Everything runs locally. No credentials, Snowflake connection, LLM, or cloud model API is used.

## Windows Command Prompt (cmd) commands

The command blocks further down use macOS/Linux (bash) syntax. On Windows, run the
equivalents below in **Command Prompt** (in VS Code: **Terminal → New Terminal**; the
prompt looks like `C:\...>`, not `PS C:\...>`). Run every command from the project
folder that contains `app.py`.

| bash (macOS / Linux) | Windows Command Prompt |
|---|---|
| `python3 -m venv .venv` | `python -m venv .venv` |
| `source .venv/bin/activate` | `.venv\Scripts\activate` |
| `export PYTHONPATH=src` or `PYTHONPATH=src python ...` | `set PYTHONPATH=src` (once per terminal), then `python ...` |
| `export SOURCE_DATASET_VERSION="..."` | `set SOURCE_DATASET_VERSION=...` (no quotes) |
| `export TRAINING_SAMPLE_ROWS=500000` | `set TRAINING_SAMPLE_ROWS=500000` |
| `scripts/name.py` | `scripts\name.py` |
| `\` at the end of a line (continue the command) | `^` |

`set` values last only for the current terminal window. In a new terminal, run
`.venv\Scripts\activate` and `set PYTHONPATH=src` again. If you see
`ModuleNotFoundError: No module named 'cluster_c'`, `PYTHONPATH` is not set.

### One-time setup

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Every new terminal

```bat
.venv\Scripts\activate
set PYTHONPATH=src
```

### Run the full workflow

```bat
python scripts\generate_sample_data.py
python scripts\prepare_data.py
python scripts\train_model.py
python scripts\prepare_ea_guidance.py
python -m pytest -q
streamlit run app.py
```

`generate_sample_data.py` is only needed to rebuild the synthetic sample
(`data\input\term_deals.csv`); skip it when using a real data extract.

### Use a new data extract

```bat
set SOURCE_DATASET_VERSION=cisco-cluster-c-20261030-v1
python scripts\prepare_data.py
python scripts\train_model.py
python -m pytest -q
streamlit run app.py
```

Optional training sample cap: `set TRAINING_SAMPLE_ROWS=500000` or `set TRAINING_SAMPLE_ROWS=all`.

### EA guidance PDFs

```bat
python scripts\generate_sample_ea_guidance.py
python scripts\prepare_ea_guidance.py
python scripts\inspect_ea_guidance.py
python scripts\inspect_ea_guidance.py --show-page-text
python scripts\validate_ea_guidance_ci.py
python -m pytest -q tests\test_guidance_streamlit_acceptance.py
```

Validate the six-rule POC release without starting Streamlit (one line in cmd):

```bat
python scripts\deploy_streamlit.py --application-version poc-v1 --expected-active-rule-ids "GROWTH-G-001,GROWTH-G-002,CUSTOMER-CV-001,CUSTOMER-CV-002,SIMPLICITY-S-001,SIMPLICITY-S-002" --warning-policy report --validate-only
```

### Useful pytest variations

```bat
python -m pytest -q
python -m pytest -v
python -m pytest -q tests\test_pipeline.py
python -m pytest -q -k synthetic
```

## First run with included sample data and model

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

The repository includes deterministic sample data, prepared features, and trained sample models, so the UI can run immediately.

## Run tests

```bash
source .venv/bin/activate
PYTHONPATH=src pytest -q
```

## Replace the sample with data

1. Read [DATA_CONTRACT.md](DATA_CONTRACT.md).
2. Rename the approved export to `term_deals.csv` and place it at `data/input/term_deals.csv`.
3. Set a traceable dataset version.
4. Prepare, train, test, and launch.

The input filename is fixed. It must remain:

```text
data/input/term_deals.csv
```

For example, if sends `term_deals.csv`, copy or rename it as follows:

```bash
cp /path/to/term_deals.csv data/input/term_deals.csv
```

`SOURCE_DATASET_VERSION` is not the filename. It is the audit label identifying the exact delivery used to prepare features and train the model. A recommended format is:

```text
-cluster-c-YYYYMMDD-vN
```

For example:

```bash
export SOURCE_DATASET_VERSION="-cluster-c-20261030-v1"
```

If provides a corrected extract on the same date, keep the filename as `term_deals.csv` and change only the version label to `v2`.

Run the complete workflow:

```bash
source .venv/bin/activate
export SOURCE_DATASET_VERSION="-cluster-c-20261030-v1"

PYTHONPATH=src python scripts/prepare_data.py
PYTHONPATH=src python scripts/train_model.py
PYTHONPATH=src pytest -q
streamlit run app.py
```

The filename remains constant for every delivery. The dataset-version label changes for each new or corrected delivery and is stored in the prepared data and model metadata for traceability.

Preparation writes model features to `data/prepared/term_model_features.parquet` using Snappy compression. The source input remains CSV, while quality checks and source profiles remain CSV for direct review.

Prepared fiscal, term, and history-count fields use compact nullable integer types; categorical model fields use dictionary encoding; and derived history averages use `float32`. Raw bookings and annualized financial amounts remain `float64`. CatBoost numeric matrices and labels use `float32` to reduce Pool memory.

By default, training uses the complete chronological training split. To reduce runtime or memory for development, set an optional deterministic sampling cap. Sampling preserves fiscal-year, term, segment, and suite representation; validation and test splits remain complete:

```bash
# Use a different training cap
export TRAINING_SAMPLE_ROWS=500000

# Explicitly use the default complete training split
export TRAINING_SAMPLE_ROWS=all
```

On Windows PowerShell, use `$env:TRAINING_SAMPLE_ROWS = "500000"` or `$env:TRAINING_SAMPLE_ROWS = "all"`.

Preparation currently records invalid critical keys, dates, term values, and financial targets as non-blocking warnings in `data/prepared/quality_checks.csv`. A missing required column still stops preparation. Training requires at least three distinct non-null term start dates. Sparse dates or missing 1-, 3-, or 5-year examples within a chronological split produce warnings because the resulting evaluation metrics are not statistically reliable.

## Optional: regenerate deterministic sample data

```bash
PYTHONPATH=src python scripts/generate_sample_data.py
PYTHONPATH=src python scripts/prepare_data.py
PYTHONPATH=src python scripts/train_model.py
```

Random seed 42 produces the same sample each time.

## What the model does

CatBoost predicts `BOOKINGS_NET_PRICE_USD_AMT`. Since model v2 it learns the annual value (`BOOKINGS_NET_PRICE_USD_AMT` ÷ term years) and multiplies each prediction back by 1, 3 or 5, so the exact "total = annual value × years" rule is built in and the model focuses on how the yearly value changes with the term. The same eligible deal is scored under `1 Year`, `3 Year`, and `5 Year` while other fields remain fixed. The scenario then mixes those estimates according to:

- Percentage of deals using flexible terms
- One-year versus five-year share
- Annual discount rate

See [MODEL_CARD.md](MODEL_CARD.md) for performance and causal limitations.

## Model comparison (different models)

`different_models()` in `src/cluster_c/model_comparison.py` trains several model families on the **same** chronological train/validation/test split, features and target as the production CatBoost, then compares them:

| Model | Family |
|---|---|
| Term × suite median | Rule-based baseline (no ML) |
| Ridge regression on log(bookings) | Linear |
| Random Forest, Extra Trees | Bagged trees |
| HistGradientBoosting, XGBoost, LightGBM | Boosted trees |
| CatBoost (current) | Production model: same settings as `model.py`, learns bookings ÷ term years and multiplies back |
| CatBoost (total-bookings target) | Previous production setup: same settings, learns total bookings directly |

Each model is scored on:

- **Accuracy** on the untouched test period: WAPE (primary), MAE, RMSE, R², MAPE, bias, share of deals within ±20%, overall and per term bucket.
- **Stability**: expanding-window backtest inside train + validation (mean and spread of WAPE). A gap smaller than the spread is treated as noise.
- **Three business lenses (PPT slide 3)**: the app's own `scenario.py` maths run on each model's 1/3/5-year scores at the default sliders — Growth (commitment change, NPV change, NPV breakeven), Customer value (average term; slider-driven, same for every model) and Simplicity (coverage volatility, renewal decision points). Saved as `business_lenses.csv`; on synthetic data each lens is also compared with the known true value.
- **Scenario fitness**: every test deal is re-scored as 1, 3 and 5 years. The model must keep 1-year < 3-year < 5-year bookings for at least 95% of deals, and the page shows the what-if answer each model would give. On synthetic data the true term effect is known (+5.25% for the default scenario), so the page also shows each model's error against it.

Run it:

```bash
PYTHONPATH=src python scripts/compare_models.py
streamlit run app.py        # then open "Model Comparison" in the left sidebar
```

Results go to `data/models/model_comparison/`. This is evaluation only; the scenario page still uses the CatBoost models in `data/models/cluster_c_term_v1/`. XGBoost and LightGBM are skipped automatically if they are not installed.
## General EA guidance PDFs

The application can apply portfolio-level EA guidance to the Growth, Customer Value, and Simplicity with Scale lenses. This is a deterministic guidance layer after scenario summarisation; it is not part of CatBoost training and does not attach documents to individual subscriptions.

For the current POC, every structurally valid rule extracted from a valid registered PDF is activated directly. Candidate-review, governance, rule-approval, and source-approval statuses are not activation gates. The older review registries and approval commands remain in the repository for future controlled-production use, but they are not required to prepare or deploy the POC guidance.

### Guidance quick start

Place one or more controlled-rule PDF files under:

```text
data/unstructured/input/
```

The repository already includes the generated POC file
`ea_general_guidance_sample_v1.pdf`.

WSL or Linux:

```bash
export PYTHONPATH=src
python scripts/prepare_ea_guidance.py
```

Windows PowerShell:

```powershell
$env:PYTHONPATH = "src"
python scripts\prepare_ea_guidance.py
```

That single command discovers the PDFs, creates or refreshes the source
manifest, calculates SHA-256 hashes, extracts page text, creates controlled
rules, activates valid rules directly for the POC, and writes the artifacts
used by Streamlit. No manual manifest, review, or approval command is required.

Optionally regenerate the included sample PDF before preparation:

```bash
python scripts/generate_sample_ea_guidance.py
python scripts/prepare_ea_guidance.py
```

The generated artifacts are:

```text
data/unstructured/manifests/guidance_sources.csv
data/unstructured/prepared/guidance_content.parquet
data/unstructured/prepared/guidance_rules.json
data/unstructured/prepared/guidance_quality.csv
data/unstructured/prepared/guidance_compatibility_report.csv
data/unstructured/prepared/guidance_preparation_summary.json
```

Changing only a guidance PDF does not require `prepare_data.py` or
`train_model.py` to be rerun.

### Optional guidance inspection

```bash
python scripts/inspect_ea_guidance.py
```

Include extracted page text in the diagnostic output with:

```bash
python scripts/inspect_ea_guidance.py --show-page-text
```

### Run Streamlit with guidance

After preparing the structured data, training the model, and preparing the PDF
guidance, start Streamlit normally:

```bash
streamlit run app.py
```

For the included POC PDF, validate the exact six-rule release without starting
Streamlit:

```bash
python scripts/deploy_streamlit.py \
  --application-version poc-v1 \
  --expected-active-rule-ids "GROWTH-G-001,GROWTH-G-002,CUSTOMER-CV-001,CUSTOMER-CV-002,SIMPLICITY-S-001,SIMPLICITY-S-002" \
  --warning-policy report \
  --validate-only
```

Remove `--validate-only` to validate the release and start Streamlit. For a
custom PDF, run `inspect_ea_guidance.py` to obtain its rule IDs and pass that
exact comma-separated list to `--expected-active-rule-ids`.

Run the optional local guidance checks with:

```bash
python scripts/validate_ea_guidance_ci.py
python -m pytest -q tests/test_guidance_streamlit_acceptance.py
```

### Guidance behavior

Guidance is evaluated after model prediction and scenario calculation. It can
change the displayed Growth, Customer Value, and Simplicity with Scale status,
but it does not modify CatBoost predictions or the underlying financial KPIs.

Rules are global unless the controlled PDF contains an optional scope:

```text
Scope: SCMS=ENTERPRISE
Scope: SALES_LEVEL_2=EMEA_NORTH
Scope: SUITE_NAME=Cisco DNA Wireless
```

A scoped rule applies only when that UI filter is exclusively selected. See
[EA_GUIDANCE_NEXT_STEPS.md](EA_GUIDANCE_NEXT_STEPS.md) for implementation details
and the controls that would be required before a production rollout.

## Revenue-stability proxy

Predicted total commitment is spread evenly over each term:

```text
1-year: commitment / 1 across year 1
3-year: commitment / 3 across years 1–3
5-year: commitment / 5 across years 1–5
```

The UI compares coverage volatility and renewal-decision points over five years. This is not revenue recognition or cash-flow logic.

## Repository structure

```text
.
├── app.py
├── pages/1_Model_Comparison.py
├── DATA_CONTRACT.md
├── MODEL_CARD.md
├── README.md
├── requirements.txt
├── data/
│   ├── input/term_deals.csv
│   ├── unstructured/
│   ├── prepared/
│   └── models/cluster_c_term_v1/
├── scripts/
│   ├── generate_sample_data.py
│   ├── prepare_data.py
│   ├── prepare_ea_guidance.py
│   ├── train_model.py
│   └── compare_models.py
├── src/cluster_c/
│   ├── synthetic.py
│   ├── preparation.py
│   ├── model.py
│   ├── model_comparison.py
│   ├── scenario.py
│   └── unstructured/
└── tests/
```

##  handoff checklist

- Confirm the exact Snowflake column spelling and subscription grain.
- Confirm whether bookings are total term commitment or another accounting measure.
- Confirm term-bucket normalization and fiscal semantics.
- Review missing-value rates and term overlap before training.
- Review test performance separately for 1-, 3-, and 5-year terms.
- Add the additional renewal and operational fields listed in the data contract.
- Replace straight-line stability proxies with -approved bookings, revenue, invoice, and cash-flow rules.
- Surface the read-only Snowflake query in the UI when CSV input is replaced, matching the PPT traceability requirement.
- Apply security, access, retention, and model-governance controls before using production data.
