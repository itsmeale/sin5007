import pandas as pd

from src.dataviz.plots import make_bar_chart_comparision
from src.evaluation.metrics import (METRICS, aggregate_metrics,
                                    clear_results_dir)
from src.train.experiment import Experiment
from src.train.models import MODELS
from src.train.scenarios import SCENARIOS

if __name__ == "__main__":
    df = pd.read_csv("data/preprocessed/HTRU_2_outliers_removed.csv")
    metrics_folder = "data/results"
    metrics_file = "data/results/metrics.csv"
    metrics = METRICS.keys()

    X = df.iloc[:, :-1]
    y = df["pulsar"]
    clear_results_dir(metrics_folder=metrics_folder)

    for scenario in SCENARIOS:
        for model in MODELS:
            (
                Experiment(
                    scenario=scenario,
                    model=model,
                )
                .run(X, y)
                .save_metrics(metrics_folder=metrics_folder)
            )

    aggregate_metrics(metrics_folder=metrics_folder, save_to=metrics_file)
    metrics_df = pd.read_csv(metrics_file)

    for metric in metrics:
        make_bar_chart_comparision(metrics_df, metric=metric)
