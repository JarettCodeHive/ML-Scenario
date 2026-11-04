# Cluster C Model Card

## Decision supported

Compare flexible 1- or 5-year terms with a 3-year reference portfolio and estimate whether the mix changes total commitment, NPV, contracted-value coverage, and renewal-decision complexity.

## Model choice

The project uses a CatBoost S-learner: one tabular regression model with `TERM_BUCKET` as the treatment feature. CatBoost is appropriate for the supplied mixture of categorical, fiscal, suite, and financial-history fields, runs locally, handles missing optional values, and can score the same deal under all three term categories.

High-cardinality identifiers `CAV_END_CUSTOMER_BU_ID` and `UCD_C4_ACCOUNT_ID` are retained for customer-history construction, split reporting, and traceability but are excluded from CatBoost inputs. This reduces categorical-statistic memory and discourages identifier memorization. Customer context is represented through leakage-safe prior-deal features instead.

Categorical-statistic growth is further constrained with `max_ctr_complexity=1` and `ctr_leaf_count_limit=100000`. This disables multi-feature categorical combinations and caps categorical table growth to reduce training time and memory. Metrics must be compared with the prior configuration because these controls can reduce model capacity.

The optimized CPU configuration uses a maximum of 400 trees at depth 6, a learning rate of 0.06, and 40-round early stopping. This replaces the previous 650-tree, depth-7 configuration. The reduced tree capacity and higher learning rate shorten training, but validation/test metrics must be compared with the previous baseline before production adoption.

By default, model fitting uses the complete chronological training split. For constrained development environments, `TRAINING_SAMPLE_ROWS` can set an exact-size deterministic training cap stratified by fiscal year, term bucket, segment, and suite. Chronological validation and test sets remain unsampled. The metadata records the source row count, sampled row count, sampling fraction, strata, and seed.

Prepared model features are stored as Snappy-compressed Parquet rather than CSV. Parquet preserves numeric/date types, supports column projection during training, and reduces repeated text parsing and storage overhead. Quality checks and source profiles remain human-readable CSV files.

Prepared fiscal fields, term fields, and customer-history counts use compact nullable integer types; categorical model inputs are dictionary encoded; and derived history averages use `float32`. Raw bookings and annualized financial values remain `float64` in prepared storage. Numeric feature matrices and training labels are converted to `float32` only at the CatBoost boundary to reduce Pool memory.

For each eligible row, the engine holds non-term fields fixed and predicts bookings three times:

```text
TERM_BUCKET = 1 Year
TERM_BUCKET = 3 Year
TERM_BUCKET = 5 Year
```

Two CatBoost models are saved locally: an MAE central model and a MultiQuantile model that produces the 10th- and 90th-percentile estimates together. The deal-level and scenario outputs continue to expose separate P10 and P90 values.

## EA guidance decision layer

General EA guidance PDFs are processed separately from the CatBoost models. Page text is deterministically converted into schema-controlled candidate rules, reviewed through a hash-bound approval registry, and evaluated only after portfolio scenario summarisation.

The guidance layer:

- Does not add PDF text or embeddings to CatBoost features.
- Does not require subscription-level document matching.
- Does not retrain or modify the saved CatBoost models.
- Allows only predefined measures, operators, outcomes, and optional portfolio scopes.
- Never executes code or arbitrary expressions extracted from a PDF.
- Requires both source-level and rule-level approval before a rule becomes active.
- Invalidates approval when the source version or SHA-256 changes.

Optional scope is restricted to `SCMS`, `SALES_LEVEL_2`, and `SUITE_NAME`. A scoped rule applies only when the corresponding UI filter exclusively selects the declared value. Unfiltered or mixed portfolios do not receive a segment-specific rule.

The included guidance PDF is synthetic and `not_approved`; it is useful for pipeline validation but cannot affect lens status. Missing or malformed guidance does not block structured model results.

This rule layer provides decision interpretation, not a learned unstructured model. No claim should be made that the PDF improves predictive accuracy. The quality and authority of displayed guidance depend on source ownership, human review, effective dates, and controlled version management.

## Synthetic evaluation

| Measure | Result |
|---|---:|
| Training rows | 3,439 |
| Validation rows | 771 |
| Untouched test rows | 790 |
| Test CatBoost WAPE | 15.40% |
| Test term-and-suite median WAPE | 47.00% |
| Test P10–P90 interval coverage | 73.16% |

Term-level test WAPE is 15.22% for 1-year, 15.30% for 3-year, and 15.60% for 5-year records.

The nominal P10–P90 range is under-covered relative to an ideal 80% interval and must be recalibrated on Cisco data.

## Causal limitation

This is an observational outcome model, not a randomized experiment or causal estimator. Historical term choice may be influenced by customer size, negotiation, risk, discounting, federal requirements, or sales judgment. Scoring a different term estimates an association conditional on available features; it does not prove that offering the term causes the predicted change.

Production analysis should add overlap diagnostics, propensity modelling, sensitivity to unobserved confounding, and—where feasible—a controlled pilot. If overlap is weak, use separate term models or a doubly robust multi-treatment estimator rather than extrapolating CatBoost.

## Financial limitation

The stability chart spreads predicted commitment evenly across the term. It is contracted-value coverage only—not GAAP revenue recognition, bookings timing, invoicing, cash flow, or Cisco accounting policy.
