import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
INPUT_PATH = DATA_DIR / "input" / "term_deals.csv"
PREPARED_DIR = DATA_DIR / "prepared"
PREPARED_FEATURE_PATH = PREPARED_DIR / "term_model_features.parquet"
MODEL_DIR = DATA_DIR / "models" / "cluster_c_term_v1"

# General EA guidance PDFs and generated guidance artifacts. These paths are
# independent of the structured term-deal preparation and CatBoost models.
UNSTRUCTURED_DIR = DATA_DIR / "unstructured"
GUIDANCE_PDF_DIR = UNSTRUCTURED_DIR / "input"
GUIDANCE_MANIFEST_PATH = UNSTRUCTURED_DIR / "manifests" / "guidance_sources.csv"
GUIDANCE_APPROVALS_PATH = UNSTRUCTURED_DIR / "manifests" / "guidance_rule_approvals.csv"
GUIDANCE_SOURCE_APPROVALS_PATH = UNSTRUCTURED_DIR / "manifests" / "guidance_source_approvals.csv"
GUIDANCE_CANDIDATE_REVIEWS_PATH = UNSTRUCTURED_DIR / "manifests" / "guidance_candidate_reviews.csv"
GUIDANCE_REVIEW_POLICY_PATH = UNSTRUCTURED_DIR / "manifests" / "guidance_review_policy.csv"
GUIDANCE_GOVERNANCE_PATH = UNSTRUCTURED_DIR / "manifests" / "guidance_governance.csv"
GUIDANCE_COMPATIBILITY_PATH = UNSTRUCTURED_DIR / "manifests" / "guidance_compatibility.csv"
GUIDANCE_PREPARED_DIR = UNSTRUCTURED_DIR / "prepared"
GUIDANCE_CONTENT_PATH = GUIDANCE_PREPARED_DIR / "guidance_content.parquet"
GUIDANCE_RULES_PATH = GUIDANCE_PREPARED_DIR / "guidance_rules.json"
GUIDANCE_QUALITY_PATH = GUIDANCE_PREPARED_DIR / "guidance_quality.csv"
GUIDANCE_COMPATIBILITY_REPORT_PATH = GUIDANCE_PREPARED_DIR / "guidance_compatibility_report.csv"
GUIDANCE_SUMMARY_PATH = GUIDANCE_PREPARED_DIR / "guidance_preparation_summary.json"
GUIDANCE_CANDIDATE_REVIEW_REPORT_PATH = GUIDANCE_PREPARED_DIR / "guidance_candidate_review_report.csv"
GUIDANCE_RELEASE_METADATA_PATH = GUIDANCE_PREPARED_DIR / "guidance_release_metadata.json"
GUIDANCE_CACHE_DIR = UNSTRUCTURED_DIR / "cache"

RANDOM_SEED = 42
SOURCE_DATASET_VERSION = os.getenv("SOURCE_DATASET_VERSION", "cluster-c-synthetic-v1-seed42")
CX_UPSELL_GROUP_VALUES = (
    "Support Services",
    "No Upsell Group",
    "Learning UG",
    "Professionsal Services",
)
XCAT_FLAG_VALUES = ("Y", "N")
ADDITIONAL_SOURCE_COLUMNS = [
    "CX_UPSELL_GROUP",
    "XCAT_FLG",
    "DV_ATTRIBUTION_CD",
]

RAW_COLUMNS = [
    "MONETIZATION_TYPE_CD",
    "TERM_BUCKET",
    "SUBSCRIPTION_REFERENCE_ID",
    "CAV_END_CUSTOMER_BU_ID",
    "CAV_END_CUSTOMER_BU_NAME",
    "UCD_C4_ACCOUNT_ID",
    "TERM_START_DATE",
    "TERM_END_DATE",
    "FISCAL_YEAR",
    "FISCAL_QUARTER_ID",
    "FISCAL_PERIOD_ID",
    "BOOKINGS_NET_PRICE_USD_AMT",
    "ANNUALIZED_NET_PRICE_USD_AMT",
    "SCMS",
    "SALES_LEVEL_2",
    "EA_CATEGORY",
    "SUITE_NAME",
    "BUSINESS_ENTITY_DESCR",
    "SUB_BUSINESS_ENTITY_DESCR",
    *ADDITIONAL_SOURCE_COLUMNS,
]
