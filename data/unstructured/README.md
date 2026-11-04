# EA Guidance Data Area

This directory currently supports general EA guidance documents only. It does not contain deal-specific contracts, recordings, images, or record-level mappings.

## Structure

```text
data/unstructured/
|-- input/
|   `-- guidance/
|       `-- pdf/          Versioned source guidance PDFs
|-- manifests/
|   |-- guidance_sources.csv
|   |-- guidance_candidate_reviews.csv
|   |-- guidance_review_policy.csv
|   |-- guidance_rule_approvals.csv
|   |-- guidance_source_approvals.csv
|   |-- guidance_governance.csv
|   `-- guidance_compatibility.csv
|-- prepared/             Generated extraction and rule artifacts
`-- cache/                Regenerable extraction cache
```

The included `ea_general_guidance_sample_v1.pdf` is synthetic test guidance. It is not Cisco-approved policy and must not be used as production decision guidance.

`guidance_sources.csv` is the source-of-truth inventory. Every guidance PDF must have a stable source ID, version, review status, and SHA-256 hash.

`guidance_rule_approvals.csv` is the human-review registry. An approval is bound to a rule ID, source ID, source version, and exact source SHA-256. Generated candidate rules cannot approve themselves.

`guidance_source_approvals.csv` retains document-level approval or rejection
evidence bound to the exact source version and SHA-256. The guarded Phase 6 command
updates it and the source manifest atomically.

`guidance_candidate_reviews.csv` records the Phase 5 source-page comparison and
is bound to a fingerprint of every extracted candidate field. It cannot override
candidate values. `guidance_review_policy.csv` records the EA owner's decision on
lens-status precedence. These reviews are prerequisites to, but not substitutes
for, individual rule approvals.

`guidance_governance.csv` records source ownership, document authority, permitted uses, retention/access expectations, and lifecycle review. A source cannot activate rules until its governance record is complete and current.

`guidance_compatibility.csv` records the reviewed onboarding method for each source. Automated assessment recommends direct controlled extraction, a controlled companion document, OCR, or rejection; activation requires a technically compatible reviewed choice.

The preparation pipeline creates the following generated files under `prepared/`:

```text
guidance_content.parquet
guidance_rules.json
guidance_quality.csv
guidance_compatibility_report.csv
guidance_preparation_summary.json
guidance_candidate_review_report.csv
guidance_release_metadata.json
```

Run `scripts/inspect_ea_guidance.py` after preparation to verify cross-artifact
counts and review source, compatibility, candidate-rule, failure, and warning
summaries. Extracted page text is hidden unless `--show-page-text` is supplied.

Generated prepared and cache files are ignored by Git. Their `.gitkeep` files retain the required directory structure.
