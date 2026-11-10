# Streamlit EA guidance acceptance evidence

- Date: 2026-11-04
- Tester: automated `pytest` using `streamlit.testing.v1.AppTest`
- Environment: repository virtual environment; test-only approved registry
- Source: `EA-TEST-APPROVED-001`, version `test-v1`
- Registry: `approved_registry.json`
- Operational source modified: no
- Screenshots: not required; assertions and rendered element values are retained in tests

## Retained scenarios

| Scenario | Expected result | Automated evidence |
|---|---|---|
| Default approved fixture | Growth favorable; structured result visible | Lens/card assertion |
| Commitment slider crosses zero | Growth changes to attention | Slider rerun assertion |
| No approved guidance | All lenses unavailable; structured result remains visible | Selectbox rerun assertion |
| Missing registry | Explicit warning; structured result remains visible | Error-state assertion |
| Unfiltered portfolio | Only global rules apply | Traceability table assertion |
| Exact SCMS selection | SCMS rule applies | Filter rerun assertion |
| Mixed SCMS selection | SCMS rule does not apply | Filter rerun assertion |
| Exact sales-level selection | Sales-level rule applies | Filter rerun assertion |
| Exact suite selection | Suite rule applies | Filter rerun assertion |
| No matching rows | User sees an explicit empty-scope message | Filter rerun assertion |
| Traceability | Rule, outcome, explanation, measures, operator, source, page, fingerprint, and generation time shown | Table/ caption assertions |

Defects found and resolved during implementation: guidance rendering and portfolio
filtering were extracted into shared production components so the acceptance
harness and the main application exercise the same lens-status, traceability, and
scope-selection code.
