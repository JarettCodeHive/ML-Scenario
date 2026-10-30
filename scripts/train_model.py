import pandas as pd

from cluster_c.config import PREPARED_DIR
from cluster_c.model import save_result, sha256, train_models

FEATURE_PATH = PREPARED_DIR / "term_model_features.csv"

if __name__ == "__main__":
    frame = pd.read_csv(FEATURE_PATH, parse_dates=["TERM_START_DATE", "TERM_END_DATE"])
    result = train_models(frame, dataset_sha256=sha256(FEATURE_PATH))
    paths = save_result(result)
    print(result.split_summary.to_string(index=False))
    print("\nMetrics:\n" + result.metrics.to_string(index=False))
    for name, path in paths.items():
        print(f"{name}: {path}")
