import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from src.dataviz.plots import make_bar_chart_comparision
from src.evaluation.metrics import METRICS, aggregate_metrics
from src.train.experiment import Experiment
from src.train.models import MODELS
from src.train.scenarios import SCENARIOS

PARALLEL_JOBS = 3


def run_experiment(experiment: Experiment, X, y, metrics_folder):
    experiment.run(X, y).save_metrics(metrics_folder=metrics_folder)


if __name__ == "__main__":
    df = pd.read_csv("data/preprocessed/HTRU_2_outliers_removed.csv")
    metrics_folder = "data/results"
    metrics_file = Path("data/results/metrics.csv")
    metrics = METRICS.keys()

    if metrics_file.exists():
        metrics_file.unlink(missing_ok=True)

    X = df.iloc[:, :-1]
    y = df["pulsar"]

    experiments = list()
    _run_experiment = partial(run_experiment, X=X, y=y, metrics_folder=metrics_folder)

    for scenario in SCENARIOS:
        for model in MODELS:
            experiments.append(Experiment(scenario=scenario, model=model))

    with Pool(processes=PARALLEL_JOBS) as p:
        p.map(_run_experiment, experiments)

    aggregate_metrics(metrics_folder=metrics_folder, save_to=metrics_file)
    metrics_df = pd.read_csv(metrics_file)

    for metric in metrics:
        make_bar_chart_comparision(metrics_df, metric=metric)
