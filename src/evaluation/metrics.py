import os
from pathlib import Path
from typing import Dict

import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)

METRICS: Dict = {
    "recall": {
        "method": recall_score,
        "array": list(),
        "mean": None,
        "ci_lower": None,
        "ci_upper": None,
    },
    "precision": {
        "method": precision_score,
        "array": list(),
        "mean": None,
        "ci_lower": None,
        "ci_upper": None,
    },
    "accuracy": {
        "method": accuracy_score,
        "array": list(),
        "mean": None,
        "ci_lower": None,
        "ci_upper": None,
    },
    "f1_score": {
        "method": f1_score,
        "array": list(),
        "mean": None,
        "ci_lower": None,
        "ci_upper": None,
    },
}


def aggregate_metrics(metrics_folder: str, save_to: str) -> pd.DataFrame:
    """Aggregate all metrics csvs in a single dataframe"""
    files = list()
    for root, _, metric_files in os.walk(metrics_folder):
        for metric_file in metric_files:
            files.append(f"{root}/{metric_file}")

    dfs = [
        pd.read_csv(file)
        for file in files
        if str(file).endswith("csv") and ("metrics.csv" not in str(file))
    ]

    df = pd.concat(dfs)
    df.to_csv(save_to, index=False)
    return df
