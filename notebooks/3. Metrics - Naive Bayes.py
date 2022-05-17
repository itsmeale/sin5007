# # Metrics - Naive Bayes

import pandas as pd

METRICS_PATH: str = "../data/results/metrics.csv"

df_metrics = pd.read_csv(METRICS_PATH)

df_metrics = df_metrics[['feature_selection', 'accuracy', 'recall', 'precision', 'f1_score']]
df_metrics


