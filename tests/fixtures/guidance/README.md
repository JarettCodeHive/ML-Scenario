# Controlled EA guidance acceptance fixtures

These fixtures exist only under `tests/` and must never be copied into the
operational manifests under `data/unstructured/`.

- `approved_registry.json` contains approved test rules covering global, `SCMS`,
  `SALES_LEVEL_2`, and `SUITE_NAME` scopes plus every supported operator.
- `lifecycle_cases.json` contains approved, rejected, future-effective, retired,
  and approval-hash-mismatch cases for activation-gate tests.
- `malformed_registry.json` is intentionally invalid JSON and verifies fail-safe
  registry loading.
- `streamlit_acceptance_app.py` exercises the production guidance rendering with
  lightweight controls instead of loading the large trained-model dataset.
- `STREAMLIT_ACCEPTANCE_EVIDENCE.md` retains the automated Phase 8 scenarios and
  results that are asserted by `tests/test_guidance_streamlit_acceptance.py`.

All names, identities, hashes, and decisions are synthetic test evidence.
