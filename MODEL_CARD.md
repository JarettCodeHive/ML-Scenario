# Cluster C Model Card

## Decision supported

Compare flexible 1- or 5-year terms with a 3-year reference portfolio and estimate whether the mix changes total commitment, NPV, contracted-value coverage, and renewal-decision complexity.

## Model choice

The project uses a CatBoost S-learner: one tabular regression model with `TERM_BUCKET` as the treatment feature. CatBoost is appropriate for the supplied mixture of categorical, identifier, fiscal, suite, and financial-history fields, runs locally, handles missing optional values, and can score the same deal under all three term categories.

For each eligible row, the engine holds non-term fields fixed and predicts bookings three times:

```text
TERM_BUCKET = 1 Year
TERM_BUCKET = 3 Year
TERM_BUCKET = 5 Year
```

Central, 10th-percentile, and 90th-percentile CatBoost models are saved locally.

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
