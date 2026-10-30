# Cisco Cluster C CSV Data Contract

The repository expects one UTF-8 comma-delimited file at `data/input/term_deals.csv`. Dates use `YYYY-MM-DD` or an ISO-8601 timestamp. Identifiers must be stable strings or integers and must not be converted to scientific notation.

The column names below reproduce the supplied Cisco schema image. If the Snowflake export uses a different suffix or spelling, rename it explicitly in the extraction query or update `src/cluster_c/config.py` under change control.

| Column | Expected type | Critical for POC | Use |
|---|---|---:|---|
| `MONETIZATION_TYPE_CD` | string | No | Categorical model feature |
| `TERM_BUCKET` | string | Yes | Must map to `1 Year`, `3 Year`, or `5 Year` |
| `SUBSCRIPTION_REFERENCE_ID` | string | Yes | Unique row/deal identifier |
| `CAV_END_CUSTOMER_BU_ID` | integer/string | Yes | Customer history and categorical feature |
| `CAV_END_CUSTOMER_BU_NAME` | string | No | UI label only; use pseudonymous names if needed |
| `UCD_C4_ACCOUNT_ID` | integer/string | No | Account-level categorical feature |
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

The screenshot marked columns nullable. The preparation layer blocks missing critical fields and lets CatBoost handle missing optional numeric values or map optional categorical values to `Unknown`.

## Grain

The required grain is one record per `SUBSCRIPTION_REFERENCE_ID`. If Cisco has multiple booking lines per subscription, aggregate them to subscription-term grain before this POC or provide a documented composite key and update the validation logic.

## Term mapping

Normalize Cisco values before preparation. Examples:

```text
12 months, 1Y, ONE_YEAR  -> 1 Year
36 months, 3Y, THREE_YEAR -> 3 Year
60 months, 5Y, FIVE_YEAR -> 5 Year
```

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
