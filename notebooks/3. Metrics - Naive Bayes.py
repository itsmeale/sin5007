# # Metrics - Naive Bayes

import pandas as pd

METRICS_PATH: str = "../data/results/metrics.csv"

df_metrics = pd.read_csv(METRICS_PATH)

df_metrics

df_metrics.sort_values(by=["model_name", "feature_selection", ], inplace=True)

df_metrics = df_metrics[["model_name", "is_balanced", 'feature_selection', 'accuracy', 'recall', 'precision', 'f1_score', "best_params",]]

df_metrics.style.highlight_max(color="#8F8")


