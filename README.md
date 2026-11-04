#  EA Cluster C ML Scenario

A standalone local proof of concept aligned to the EA Lever Modelling Scope v4:

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
- pypdf
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

CatBoost predicts `BOOKINGS_NET_PRICE_USD_AMT`. The same eligible deal is scored under `1 Year`, `3 Year`, and `5 Year` while other fields remain fixed. The scenario then mixes those estimates according to:

- Percentage of deals using flexible terms
- One-year versus five-year share
- Annual discount rate

See [MODEL_CARD.md](MODEL_CARD.md) for performance and causal limitations.

## General EA guidance PDFs

The application can apply reviewed, portfolio-level EA guidance to the Growth, Customer Value, and Simplicity with Scale lenses. This is a deterministic guidance layer after scenario summarisation; it is not part of CatBoost training and does not attach documents to individual subscriptions.

Guidance sources and human reviews are stored under:

```text
data/unstructured/input/guidance/pdf/
data/unstructured/manifests/guidance_sources.csv
data/unstructured/manifests/guidance_rule_approvals.csv
data/unstructured/manifests/guidance_source_approvals.csv
data/unstructured/manifests/guidance_candidate_reviews.csv
data/unstructured/manifests/guidance_review_policy.csv
data/unstructured/manifests/guidance_governance.csv
data/unstructured/manifests/guidance_compatibility.csv
```

Prepare guidance independently:

```bash
PYTHONPATH=src python scripts/prepare_ea_guidance.py
PYTHONPATH=src python scripts/inspect_ea_guidance.py
PYTHONPATH=src python scripts/review_ea_guidance.py --initialize
```

Register a new real source with the guarded onboarding command:

```bash
PYTHONPATH=src python scripts/register_ea_guidance.py --help
```

Use `--dry-run` first. Registration requires owners, reviewers, permitted-use confirmation, retention/access terms, and a reviewed onboarding method; it refuses duplicate IDs and never overwrites an existing PDF.

This validates source paths and SHA-256 hashes, extracts PDF pages, creates controlled candidate rules, applies human-approval gates, and writes:

```text
data/unstructured/prepared/guidance_content.parquet
data/unstructured/prepared/guidance_rules.json
data/unstructured/prepared/guidance_quality.csv
data/unstructured/prepared/guidance_compatibility_report.csv
data/unstructured/prepared/guidance_preparation_summary.json
data/unstructured/prepared/guidance_candidate_review_report.csv
```

The inspection command checks that all generated artifact counts agree and shows
source, compatibility, candidate-rule, failure, and warning summaries. It does
not display PDF page text unless `--show-page-text` is explicitly supplied.

Candidate review is hash-bound to the source and every extracted rule field.
Reviewers confirm page comparison, ambiguity handling, measure/operator, scope,
and lens-status precedence. Corrections must be made in the controlled source and
regenerated; the review registry cannot override extracted rule values. Phase 5
confirmation is required for activation but remains separate from Phase 6 approval.

Record Phase 6 decisions through the guarded command—use `--dry-run` first:

```bash
PYTHONPATH=src python scripts/record_ea_guidance_approval.py --help
```

Approved rule reviewers must match the governance registry. Source decisions are
written atomically to the source manifest and hash-bound source-approval registry.
Synthetic sources are prevented from being approved.

Controlled acceptance fixtures are isolated under `tests/fixtures/guidance/`.
They cover every supported rule operator and scope, approved/rejected decisions,
future and retired sources, hash mismatch, missing measures, and malformed or
missing registries without approving the operational synthetic PDF. Run:

```bash
PYTHONPATH=src python -m pytest -q tests/test_guidance_acceptance_fixtures.py
```

The Streamlit guidance layer has a lightweight acceptance harness that calls the
same production lens and traceability renderers without loading the large model
dataset. Run:

```bash
PYTHONPATH=src python -m pytest -q tests/test_guidance_streamlit_acceptance.py
```

Retained evidence is stored in
`tests/fixtures/guidance/STREAMLIT_ACCEPTANCE_EVIDENCE.md`.

GitHub Actions runs the EA guidance quality gate from
`.github/workflows/ea-guidance-ci.yml`. Locally, run the same controls with:

```bash
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python scripts/prepare_ea_guidance.py
PYTHONPATH=src python scripts/validate_ea_guidance_ci.py
```

Warnings are reported by default. Set `GUIDANCE_CI_WARNING_POLICY=fail` for a
strict release gate. Generated guidance artifacts are recreated and uploaded as
CI evidence, not committed.

Build a deployable guidance release only with an explicit expected rule set:

```bash
PYTHONPATH=src python scripts/build_ea_guidance_release.py \
  --application-version <version> \
  --expected-active-rule-ids <RULE-1,RULE-2> \
  --warning-policy fail
```

Use `none` explicitly when no active rules are expected. The builder stages and
validates artifacts, verifies exact active IDs, records release metadata, and
restores the prior prepared directory if publication fails. Start the application
through the guarded deployment command:

```bash
PYTHONPATH=src python scripts/deploy_streamlit.py \
  --application-version <version> \
  --expected-active-rule-ids <RULE-1,RULE-2> \
  --warning-policy fail
```

A rule becomes active only when its exact source version/hash and its individual approval record are approved with reviewer evidence, and its source has complete ownership, authority, permitted-use, retention/access, and current lifecycle governance. The included synthetic PDF remains unapproved and produces no active rules.

Rules are global unless the controlled PDF contains an optional scope such as:

```text
Scope: SCMS=ENTERPRISE
Scope: SALES_LEVEL_2=EMEA_NORTH
Scope: SUITE_NAME=Cisco DNA Wireless
```

A scoped rule applies only when that UI filter is exclusively selected. Guidance changes do not require `prepare_data.py` or `train_model.py` to be rerun. See [EA_GUIDANCE_NEXT_STEPS.md](EA_GUIDANCE_NEXT_STEPS.md) for implementation and governance details.

Compatibility assessment classifies each source as controlled text, free-form text, OCR-required, or unsupported. The reviewed onboarding choice must be recorded as one of:

```text
direct_controlled_pdf
controlled_companion_pdf
manual_rule_registry
ocr_required
rejected
```

Only `direct_controlled_pdf` is currently eligible for automatic rule activation. Other choices document the required remediation or onboarding path.

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
│   ├── unstructured/
│   ├── prepared/
│   └── models/cluster_c_term_v1/
├── scripts/
│   ├── generate_sample_data.py
│   ├── prepare_data.py
│   ├── prepare_ea_guidance.py
│   └── train_model.py
├── src/cluster_c/
│   ├── synthetic.py
│   ├── preparation.py
│   ├── model.py
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
