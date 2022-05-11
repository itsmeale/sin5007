from pprint import pprint
from typing import Dict, List

import numpy as np
import pandas as pd
import statsmodels.stats.api as stats
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

METRICS: Dict = {
    "recall": {
        "method": recall_score,
        "array": list(),
        "mean": None,
        "ic_min": None,
        "ic_max": None,
    },
    "precision": {
        "method": precision_score,
        "array": list(),
        "mean": None,
        "ic_min": None,
        "ic_max": None,
    },
    "accuracy": {
        "method": accuracy_score,
        "array": list(),
        "mean": None,
        "ic_min": None,
        "ic_max": None,
    },
    "f1": {
        "method": f1_score,
        "array": list(),
        "mean": None,
        "ic_min": None,
        "ic_max": None,
    },
}


def __get_positives_and_negatives(y):
    """Calcula proporcao de positivos e negativos com base no vetor alvo y"""
    t = len(y)
    p = len(np.where(y == 1)[0])
    n = t - p
    return t, p, n


def summary_metric_array(metric_array: List):
    """Calcula media e intervalo de confianca par ao vetor metric_array"""
    mean = np.mean(metric_array)
    ci = stats.DescrStatsW(metric_array).tconfint_mean()
    return mean, ci


def print_dataset_balance(y):
    t, positive, negative = __get_positives_and_negatives(y)

    print(
        f"{positive} positivas e {negative} negativas "
        f"({positive/t:.2%} x {negative/t:.2%})"
    )


def cross_validate(estimator, df: pd.DataFrame, k: int, metrics: Dict):
    # separa variaveis independentes X da variavel alvo y
    X = df.iloc[:, :-1]
    y = df["pulsar"]

    print_dataset_balance(y)
    kfold = StratifiedKFold(n_splits=k)

    # para cada fold
    for idx, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
        X_train, X_test = X.iloc[train_idx, :], X.iloc[test_idx, :]
        y_train, y_test = y[train_idx], y[test_idx]

        # ajuste e predicoes
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)

        # calcula metricas de desempenho
        for metric in metrics.keys():
            value = metrics[metric]["method"](y_test, y_pred)
            metrics[metric]["array"].append(value)

    for metric in metrics.keys():
        # calcula media e intervalo de confianca (95%) para cada metrica
        mean, (ic_min, ic_max) = summary_metric_array(metrics[metric]["array"])
        metrics[metric]["mean"] = mean
        metrics[metric]["ic_min"] = ic_min
        metrics[metric]["ic_max"] = ic_max

    return metrics


if __name__ == "__main__":
    from sklearn.naive_bayes import GaussianNB

    df = pd.read_csv("data/preprocessed/HTRU_2_outliers_removed.csv")
    pprint(cross_validate(estimator=GaussianNB(), df=df, k=10, metrics=METRICS))
