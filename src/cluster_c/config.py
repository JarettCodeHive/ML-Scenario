import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
INPUT_PATH = DATA_DIR / "input" / "term_deals.csv"
PREPARED_DIR = DATA_DIR / "prepared"
MODEL_DIR = DATA_DIR / "models" / "cluster_c_term_v1"
RANDOM_SEED = 42
SOURCE_DATASET_VERSION = os.getenv("SOURCE_DATASET_VERSION", "cluster-c-synthetic-v1-seed42")

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
]
