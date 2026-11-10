# EA Guidance Data Area

This directory currently supports general EA guidance documents only. It does not contain deal-specific contracts, recordings, images, or record-level mappings.

## Structure

```text
data/unstructured/
|-- input/
|   `-- *.pdf             Source guidance PDFs placed by the user
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

The included `ea_general_guidance_sample_v1.pdf` is generated POC guidance. It is not Cisco policy and should be replaced before production use.

Run the complete POC preparation flow with:

```bash
PYTHONPATH=src python scripts/prepare_ea_guidance.py
```

The command recursively discovers PDFs under `input/`, generates
`guidance_sources.csv`, calculates source hashes, extracts page text, creates
controlled rules, activates valid rules directly, and writes the prepared
artifacts. Users do not need to edit a manifest or run review/approval commands.

The review, approval, governance, and compatibility registries are retained only
for possible future production governance. They do not gate POC activation.

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
