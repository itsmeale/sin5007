from typing import Dict, List

import numpy as np
import pandas as pd
import statsmodels.stats.api as stats
from sklearn.model_selection import StratifiedKFold


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


def cross_validate(data: pd.DataFrame, estimator, k: int, metrics: Dict):
    # separa variaveis independentes X da variavel alvo y
    X = data.iloc[:, :-1]
    y = data["pulsar"]

    print_dataset_balance(y)
    kfold = StratifiedKFold(n_splits=k)

    # TODO: adaptar funcao para rodar os k folds para cada conjunto
    # de hiperparametros

    # para cada fold
    for idx, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
        X_train, X_test = X.iloc[train_idx, :], X.iloc[test_idx, :]
        y_train, y_test = y[train_idx], y[test_idx]

        # ajuste e predicoes
        clf = estimator()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # calcula metricas de desempenho
        for metric in metrics.keys():
            value = metrics[metric]["method"](y_test, y_pred)
            metrics[metric]["array"].append(value)

    for metric in metrics.keys():
        # calcula media e intervalo de confianca (95%) para cada metrica
        mean, (ci_lower, ci_upper) = summary_metric_array(metrics[metric]["array"])
        metrics[metric]["mean"] = mean
        metrics[metric]["ci_lower"] = ci_lower
        metrics[metric]["ci_upper"] = ci_upper

    # TODO: adaptar o retorno para ser o melhor conjunto de hiperparametros selecionado
    # com base em uma metrica especifica escolhida por parametro da funcao
    return metrics
