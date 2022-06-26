# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # NB

import pandas as pd

df = pd.read_csv("../data/results/metrics.csv")

svm_metrics = df[df.model_name == "Naive Bayes"]

_svm_metrics = pd.concat(
    [
        svm_metrics[["scenario_name", "accuracy", "precision", "recall", "f1_score"]],
        svm_metrics["best_params"].map(eval).apply(pd.Series)
    ],
    axis=1
)

(
    _svm_metrics
    .sort_values(by=["scenario_name"])
    .style
    .highlight_max(subset=["accuracy", "precision", "recall", "f1_score"], color="#8F8")
)


