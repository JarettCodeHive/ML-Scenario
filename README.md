#  EA Cluster C ML Scenario

A standalone local proof of concept aligned to the  EA Lever Modelling Scope v4:

> **Cluster:** C — Term and Renewal  
> **What-if scenario:** Flexible 1- or 5-year terms against 3-year  
> **Core question:** Does term flexibility grow commitments or destabilise revenue?  
> **PPT register:** Wave 1 · Scenario 8 · Type 1

The project contains only this scenario: synthetic -shaped data, quality checks, point-in-time feature preparation, local CatBoost training, deterministic portfolio calculations, and a one-page Streamlit decision view.

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
- Streamlit
- Plotly
- pytest

Everything runs locally. No credentials, Snowflake connection, LLM, or cloud model API is used.

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

Preparation currently records invalid critical keys, dates, term values, and financial targets as non-blocking warnings in `data/prepared/quality_checks.csv`. A missing required column still stops preparation. Training also fails unless historical 1-, 3-, and 5-year examples exist in every chronological split.

## Optional: regenerate deterministic sample data

```bash
PYTHONPATH=src python scripts/generate_sample_data.py
PYTHONPATH=src python scripts/prepare_data.py
PYTHONPATH=src python scripts/train_model.py
```

Random seed 42 produces the same sample each time.

## What the model does

CatBoost predicts `BOOKINGS_NET_PRICE_USD_AMT`. The same eligible deal is scored under `1 Year`, `3 Year`, and `5 Year` while other fields remain fixed. The scenario then mixes those estimates according to:

- Percentage of deals using flexible terms
- One-year versus five-year share
- Annual discount rate

See [MODEL_CARD.md](MODEL_CARD.md) for performance and causal limitations.

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
├── DATA_CONTRACT.md
├── MODEL_CARD.md
├── README.md
├── requirements.txt
├── data/
│   ├── input/term_deals.csv
│   ├── prepared/
│   └── models/cluster_c_term_v1/
├── scripts/
│   ├── generate_sample_data.py
│   ├── prepare_data.py
│   └── train_model.py
├── src/cluster_c/
│   ├── synthetic.py
│   ├── preparation.py
│   ├── model.py
│   └── scenario.py
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
- Apply  security, access, retention, and model-governance controls before using production data.
