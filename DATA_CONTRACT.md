# Cisco Cluster C CSV Data Contract

The repository expects one UTF-8 comma-delimited file at `data/input/term_deals.csv`. Dates use `YYYY-MM-DD` or an ISO-8601 timestamp. Identifiers must be stable strings or integers and must not be converted to scientific notation.

The column names below reproduce the supplied Cisco schema image. If the Snowflake export uses a different suffix or spelling, rename it explicitly in the extraction query or update `src/cluster_c/config.py` under change control.

| Column | Expected type | Critical for POC | Use |
|---|---|---:|---|
| `MONETIZATION_TYPE_CD` | string | No | Categorical model feature |
| `TERM_BUCKET` | string | Yes | Must map to `1 Year`, `3 Year`, or `5 Year` |
| `SUBSCRIPTION_REFERENCE_ID` | string | Yes | Unique row/deal identifier |
| `CAV_END_CUSTOMER_BU_ID` | integer/string | Yes | Customer history, split reporting, and traceability; excluded from model features |
| `CAV_END_CUSTOMER_BU_NAME` | string | No | UI label only; use pseudonymous names if needed |
| `UCD_C4_ACCOUNT_ID` | integer/string | No | Account-level traceability; excluded from model features |
| `TERM_START_DATE` | date/timestamp | Yes | Chronological split and history cutoff |
| `TERM_END_DATE` | date/timestamp | Yes | Term validation |
| `FISCAL_YEAR` | integer | No | Numeric model feature |
| `FISCAL_QUARTER_ID` | integer 1–4 | No | Numeric model feature |
| `FISCAL_PERIOD_ID` | integer 1–12 | No | Numeric model feature |
| `BOOKINGS_NET_PRICE_USD_AMT` | decimal | Yes | ML target: total committed bookings amount |
| `ANNUALIZED_NET_PRICE_USD_AMT` | decimal | No | Prior-customer-value history; never uses the current row as an input |
| `SCMS` | string | No | Segment/filter and categorical feature |
| `SALES_LEVEL_2` | string | No | Theatre/sales filter and categorical feature |
| `EA_CATEGORY` | string | No | Full/Partial Commit or Cisco category |
| `SUITE_NAME` | string | No | Suite filter and categorical feature |
| `BUSINESS_ENTITY_DESCR` | string | No | Categorical model feature |
| `SUB_BUSINESS_ENTITY_DESCR` | string | No | Categorical model feature |

The screenshot marked columns nullable. For the current exploratory workflow, the preparation layer reports missing critical fields in `data/prepared/quality_checks.csv` without stopping preparation. Missing required columns remain blocking. CatBoost handles missing optional numeric values and maps optional categorical values to `Unknown`.

## Grain

The required grain is one record per `SUBSCRIPTION_REFERENCE_ID`. If Cisco has multiple booking lines per subscription, aggregate them to subscription-term grain before this POC or provide a documented composite key and update the validation logic.

## Term mapping

Normalize Cisco values before preparation. Examples:

```text
12 months, 1Y, ONE_YEAR  -> 1 Year
36 months, 3Y, THREE_YEAR -> 3 Year
60 months, 5Y, FIVE_YEAR -> 5 Year
```

The preparation step performs these common mappings automatically. Unrecognized non-null labels remain visible in the quality-check report.

`TERM_END_DATE - TERM_START_DATE` must agree with `TERM_BUCKET` within five days, allowing leap years.

## Point-in-time customer history

The preparation step derives these features only from earlier rows for the same customer:

- Prior deal count
- Prior average annualized net price
- Prior average bookings

The current deal and future deals are excluded. A test changes the current target and proves its prior-history feature remains unchanged.

## Additional data required for the full PPT business case

The supplied columns support commitment and contracted-value stability modelling. The PPT implementation plan additionally requests:

- Renewal date, status, and outcome
- Early-renewal date and acceptance
- Price and discount by contract year
- Cancellation and churn events
- Expansion and suite attach at renewal
- Federal/customer-type indicator where permitted
- Payment and invoice frequency
- Approval touches and operational cost

Until those fields are available, the UI must not claim renewal improvement, churn reduction, cash-flow impact, invoice cost, or causal customer acceptance.

## General EA guidance PDF contract

General EA guidance is a separate portfolio-level input and is not joined to `term_deals.csv`. It does not require `SUBSCRIPTION_REFERENCE_ID` or another deal identifier.

Text-based guidance PDFs must be placed under:

```text
data/unstructured/input/guidance/pdf/
```

The preferred onboarding path is `scripts/register_ea_guidance.py`. It generates a versioned filename, prevents overwrite, calculates SHA-256, and appends coordinated source, governance, and compatibility records. A newly registered real source always starts as `in_review`.

Every source requires an entry in `data/unstructured/manifests/guidance_sources.csv` with:

```text
source_id
source_path
source_type
guidance_type
title
version
effective_date
review_status
is_synthetic
sha256
```

Source paths must be repository-relative and remain inside the approved PDF directory. The SHA-256 must match the file exactly. Changing a PDF requires a new hash and review; previous hash-bound rule approvals no longer match.

Allowed source review statuses are:

```text
not_approved
in_review
approved
retired
rejected
```

Candidate rules require a matching record in `data/unstructured/manifests/guidance_rule_approvals.csv`. An approved rule must record the reviewer and review timestamp. A rule is active only when the source and rule are both approved, the hashes and versions match, and the source effective date has been reached.

Every source also requires a record in `data/unstructured/manifests/guidance_governance.csv`. Production activation requires:

- Business owner
- Technical/model owner
- Authorized rule reviewer and backup reviewer
- Source repository and governance contact
- Approved authority status with confirmer and timestamp
- Explicit permission for storage, text extraction, rule derivation, and Streamlit reference
- Retention policy and access requirements
- Valid retirement/supersession details
- Positive review frequency and current authority review

Draft and in-review sources may be extracted for review, but incomplete governance prevents their rules from becoming active.

Every source also requires an onboarding decision in `data/unstructured/manifests/guidance_compatibility.csv`:

```text
source_id
selected_onboarding_method
assessed_by
assessed_at_utc
assessment_notes
```

Allowed methods are:

```text
direct_controlled_pdf
controlled_companion_pdf
manual_rule_registry
ocr_required
rejected
```

The automated compatibility report records PDF readability, page count, extracted character count, controlled-rule count, format classification, recommended method, reviewed selection, and activation compatibility. The current runtime activates automatically extracted rules only from reviewed `direct_controlled_pdf` sources. Free-form documents require a controlled companion or separately governed manual rule process; image-only documents require a future OCR implementation.

Controlled PDF rules support global scope or one optional portfolio filter:

```text
Scope: global
Scope: SCMS=ENTERPRISE
Scope: SALES_LEVEL_2=EMEA_NORTH
Scope: SUITE_NAME=Cisco DNA Wireless
```

No other scope columns are accepted. Segment scope filters portfolio interpretation and does not associate guidance with individual subscriptions.

The current extractor supports normal text-based PDFs. Encrypted, corrupt, empty, and image-only PDFs fail preparation with an explicit quality status. OCR is not currently enabled.
