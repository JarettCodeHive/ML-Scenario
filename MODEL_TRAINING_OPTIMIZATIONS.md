# Model Training Optimization Options

## Current baseline

The current large test dataset contains approximately 3.8 million rows. On the current machine, CatBoost training with `thread_count=2` took approximately 2.7 hours and reached about 95% memory usage.

Current model configuration:

```text
Models trained:       2 (point, MultiQuantile P10/P90)
Iterations:           400
Tree depth:           6
Learning rate:        0.06
Early stopping:       40 rounds
CatBoost threads:     2
Prepared data format: Snappy-compressed Parquet
```

Because memory is nearly exhausted, increasing the thread count should not be the first optimization. Memory duplication and categorical-processing costs should be reduced before testing additional threads.

## Optimization summary

| ID | Optimization | Primary benefit | Model behavior change | Priority |
|---|---|---|---:|---:|
| OPT-01 | Remove the unused test CatBoost Pool | Lower peak memory | No | High |
| OPT-02 | Reduce full DataFrame and split copies | Lower peak memory | No | High |
| OPT-03 | Make thread count configurable | Easier controlled benchmarking | No | High |
| OPT-04 | Add restartable CatBoost snapshots | Recovery after interruption | No | High |
| OPT-05 | Remove high-cardinality identifier features | Faster training and lower memory | Yes | High |
| OPT-06 | Restrict categorical feature combinations | Faster categorical processing | Yes | High |
| OPT-07 | Replace P10/P90 models with MultiQuantile | Train two models instead of three | Yes | Medium |
| OPT-08 | Reduce depth and iteration count | Faster training | Yes | Medium |
| OPT-09 | Train on a representative sample | Substantially faster development | Yes | High |
| OPT-10 | Convert prepared data from CSV to Parquet | Faster loading and lower conversion cost | No | Medium |
| OPT-11 | Apply explicit compact data types | Lower DataFrame memory | No | Medium |
| OPT-12 | Add a CatBoost RAM limit | Limit some categorical-statistic memory | Potentially | Medium |
| OPT-13 | Evaluate GPU training | Potentially much faster training | Potentially | Low/conditional |

## Detailed options

### OPT-01: Remove the unused test Pool

The current code creates CatBoost pools for train, validation, and test splits:

```python
pools = {
    name: Pool(...)
    for name, part in splits.items()
}
```

Only the train and validation pools are passed to `fit()`. Test predictions are created later from a DataFrame, so the test Pool consumes memory without being used.

Recommended change:

```python
pools = {
    name: Pool(
        matrix(splits[name]),
        label=splits[name][TARGET].astype(float),
        cat_features=CATEGORICAL_FEATURES,
        feature_names=FEATURES,
    )
    for name in ("train", "validation")
}
```

Expected result:

- Lower peak memory during training.
- No change to model behavior or metrics.
- More headroom for a later thread-count benchmark.

### OPT-02: Reduce DataFrame and split copies

The training path currently creates several large copies:

- A complete copy inside `chronological_split()`.
- Separate copied train, validation, and test frames.
- Feature-matrix copies inside `matrix()`.
- CatBoost's internal Pool representation.

Possible changes:

- Parse dates before calling `chronological_split()` and avoid copying the complete frame.
- Keep row masks or indices instead of immediately copying every split.
- Build the train and validation pools, then release temporary matrices.
- Retain only the columns needed for evaluation in the test split.
- Explicitly release the input frame before fitting when it is no longer required.

Expected result:

- Lower peak memory.
- No intended model behavior change.
- Implementation requires careful testing to avoid view/copy bugs.

### OPT-03: Make thread count configurable

The thread count is currently hardcoded in `src/cluster_c/model.py`.

Recommended configuration:

```python
thread_count=int(os.getenv("CATBOOST_THREAD_COUNT", "2"))
```

Example:

```bash
CATBOOST_THREAD_COUNT=2 PYTHONPATH=src python scripts/train_model.py
```

On Windows PowerShell:

```powershell
$env:CATBOOST_THREAD_COUNT = "2"
$env:PYTHONPATH = "src"
python scripts\train_model.py
```

Recommended benchmark sequence after memory optimizations:

1. Two threads.
2. Four threads if peak memory stays below approximately 80–85%.
3. Do not use all available cores on the current low-memory machine.

CatBoost documents that CPU `thread_count` affects execution speed and not model results: <https://catboost.ai/docs/en/references/training-parameters/performance>

### OPT-04: Add restartable CatBoost snapshots

Snapshots do not make training faster, but they reduce the cost of a WSL, Windows, or process crash.

Example configuration:

```python
allow_writing_files=True,
save_snapshot=True,
snapshot_file="data/models/cluster_c_term_v1/point_training.snapshot",
snapshot_interval=300,
```

Each model must use a different snapshot file:

```text
point_training.snapshot
lower_training.snapshot
upper_training.snapshot
```

Expected result:

- Interrupted training can resume.
- Small amount of additional disk I/O.
- No intended model behavior change.

### OPT-05: Remove high-cardinality identifier features

The following categorical features can contain hundreds of thousands or millions of distinct values:

```text
CAV_END_CUSTOMER_BU_ID
UCD_C4_ACCOUNT_ID
```

The generated large dataset contains approximately one million customers. CatBoost must calculate categorical statistics for these identifiers, which can substantially increase training time and memory.

The preparation pipeline already provides customer-history features:

```text
CUSTOMER_PRIOR_DEAL_COUNT
CUSTOMER_PRIOR_AVG_ANNUALIZED_PRICE
CUSTOMER_PRIOR_AVG_BOOKINGS
```

Evaluation procedure:

1. Train a baseline on the same representative sample.
2. Remove the two identifier features.
3. Compare validation and test WAPE, bias, interval coverage, runtime, and peak memory.
4. Remove them permanently only if performance remains acceptable.

Expected result:

- Potentially large training-time and memory reduction.
- May improve generalization by reducing identifier memorization.
- Changes model behavior and requires an accuracy comparison.

### OPT-06: Restrict categorical feature combinations

Candidate CatBoost settings:

```python
max_ctr_complexity=1,
ctr_leaf_count_limit=100_000,
```

`max_ctr_complexity=1` prevents CatBoost from creating combinations of multiple categorical features. `ctr_leaf_count_limit` limits categorical-statistic table growth.

Expected result:

- Faster training and reduced memory, especially with high-cardinality features.
- May reduce predictive accuracy.
- Must be tested on a representative sample before full training.

CatBoost training-parameter reference: <https://catboost.ai/docs/en/references/training-parameters/>

### OPT-07: Replace separate P10 and P90 models with MultiQuantile

The current workflow trains:

1. One MAE point model.
2. One P10 quantile model.
3. One P90 quantile model.

CatBoost supports a CPU `MultiQuantile` objective:

```python
loss_function="MultiQuantile:alpha=0.1,0.9"
```

This could reduce the workflow to:

1. One point model.
2. One multi-quantile interval model.

Expected result:

- Approximately 25–35% lower total training time, depending on convergence.
- Fewer saved model files.
- Requires changes to training, saving, loading, prediction shapes, and tests.
- Interval coverage must be compared with the existing two-model approach.

CatBoost MultiQuantile documentation: <https://catboost.ai/docs/en/concepts/loss-functions-regression>

### OPT-08: Reduce tree depth and iteration count

Previous configuration:

```python
iterations=650
depth=7
learning_rate=0.04
early_stopping_rounds=70
```

Implemented configuration:

```python
iterations=400
depth=6
learning_rate=0.06
early_stopping_rounds=40
```

Recommended evaluation:

- Record the best iteration and duration for all three existing models.
- Reduce the iteration maximum if models consistently stop well below 650.
- Compare accuracy before and after reducing depth.

Expected result:

- Potentially significant runtime reduction.
- Depth 6 creates half as many leaves as depth 7 in a symmetric tree.
- Changes model capacity and requires metric comparison.

### OPT-09: Train on a representative sample

Training on every generated row may not add useful information because many generated rows follow the same deterministic patterns.

Suggested dataset sizes:

```text
Development benchmark:  250,000–500,000 rows
Candidate evaluation:   1,000,000 rows
Final approved run:      Full dataset, only if metrics justify it
```

Implemented behavior:

```text
Default training cap: none (complete training split)
Sampling scope:       Chronological training split only
Unsampled data:       Validation and test splits
Strata:               Fiscal year, term bucket, SCMS, suite
Seed:                 Training random seed
```

Configure the cap with `TRAINING_SAMPLE_ROWS`:

```bash
TRAINING_SAMPLE_ROWS=500000 PYTHONPATH=src python scripts/train_model.py
TRAINING_SAMPLE_ROWS=all PYTHONPATH=src python scripts/train_model.py
```

The sample should preserve:

- Chronological periods.
- 1-, 3-, and 5-year term proportions.
- Sales segments.
- Suites and business entities.
- Customer-size distribution.

Expected result:

- Largest predictable runtime reduction.
- Requires a deterministic, documented sampling method.
- Final model metrics must clearly identify the sampled dataset.

### OPT-10: Convert prepared data to Parquet

The prepared feature CSV is approximately 1.1 GB. CSV requires text parsing and type inference on every load.

Parquet benefits:

- Faster loading.
- Smaller storage footprint.
- Preserved numeric and date types.
- Ability to read selected columns.

This primarily improves loading and preprocessing rather than CatBoost fitting time. It is implemented with PyArrow and Snappy compression across preparation, training, and app loading.

### OPT-11: Apply explicit compact data types

Implemented prepared-data types include:

```text
FISCAL_YEAR                       nullable Int16
FISCAL_QUARTER_ID                 nullable Int32
FISCAL_PERIOD_ID                  nullable Int32
TERM_START_MONTH                  nullable Int8
TERM_YEARS                        nullable Int8
CUSTOMER_PRIOR_DEAL_COUNT         nullable Int32
derived prior-value averages      float32
raw financial values              float64
categorical model fields          dictionary encoded
CatBoost numeric matrix/labels    float32
```

Raw `BOOKINGS_NET_PRICE_USD_AMT` and `ANNUALIZED_NET_PRICE_USD_AMT` remain `float64` in prepared storage. Conversion to `float32` occurs only for CatBoost's numeric input matrix and training labels, where features are quantized internally.

### OPT-12: Apply a CatBoost RAM limit

Candidate setting:

```python
used_ram_limit="8gb",
```

This is not a hard process-memory limit. CatBoost documents that it primarily affects memory used for categorical-statistic calculations and may not always keep total usage under the requested value.

Reference: <https://catboost.ai/docs/en/references/training-parameters/performance>

Expected result:

- May reduce categorical-statistic memory.
- Can make training slower.
- Does not replace DataFrame and Pool memory optimization.

### OPT-13: Evaluate GPU training

GPU training is conditional on having a supported NVIDIA GPU and sufficient GPU memory. CatBoost supports MAE and Quantile objectives on GPU, while MultiQuantile is documented as CPU-only.

Candidate settings:

```python
task_type="GPU",
devices="0",
```

Expected result:

- Potentially substantial runtime reduction.
- Requires compatible NVIDIA drivers and GPU memory.
- CPU and GPU training may not produce bit-for-bit identical results.
- Must be benchmarked separately from the CPU configuration.

## Recommended implementation phases

### Phase 1: Low-risk memory and resilience changes

- OPT-01: Remove unused test Pool.
- OPT-02: Reduce DataFrame copies.
- OPT-03: Make thread count configurable.
- OPT-04: Add restartable snapshots.

These changes should be completed before increasing the thread count.

### Phase 2: Controlled model-performance experiments

- OPT-05: Remove high-cardinality identifier features.
- OPT-06: Restrict categorical combinations.
- OPT-08: Reduce depth and iterations.
- OPT-09: Add representative sampling.

Each experiment should use the same dataset and report runtime, peak memory, WAPE, bias, and interval coverage.

### Phase 3: Larger workflow changes

- OPT-07: Use MultiQuantile for P10/P90.
- OPT-10: Move prepared data to Parquet.
- OPT-11: Apply compact data types.
- OPT-13: Evaluate GPU training if supported.

## Acceptance criteria

Every selected optimization should be evaluated using:

```text
Preparation/loading time
Pool-construction time
Point-model training time
P10/P90 or MultiQuantile training time
Peak resident memory
Best iteration per model
Validation and test WAPE
Validation and test bias
P10–P90 interval coverage
Saved model size
```

An optimization should not be accepted solely because it is faster. Model metrics, chronological split integrity, traceability, and scenario-output compatibility must remain acceptable.

## Selection checklist

Mark the options to implement:

```text
[x] OPT-01 Remove unused test Pool
[x] OPT-02 Reduce DataFrame copies
[x] OPT-03 Make thread count configurable
[ ] OPT-04 Add restartable snapshots
[x] OPT-05 Remove high-cardinality IDs
[x] OPT-06 Restrict categorical combinations
[x] OPT-07 Use MultiQuantile
[x] OPT-08 Reduce depth and iterations
[x] OPT-09 Add representative sampling
[x] OPT-10 Use Parquet
[x] OPT-11 Apply compact data types
[ ] OPT-12 Set CatBoost RAM limit
[ ] OPT-13 Evaluate GPU training
```
