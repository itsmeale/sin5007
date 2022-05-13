from typing import Dict, List

import numpy as np
import pandas as pd
import statsmodels.stats.api as stats
from loguru import logger
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm


def summary_metric_array(metric_array: List):
    """Calcula media e intervalo de confianca par ao vetor metric_array"""
    mean = np.mean(metric_array)
    ci = stats.DescrStatsW(metric_array).tconfint_mean()
    return mean, ci


def __run_kfolds(clf, folds, X, y, criterion, metrics):
    _metrics = metrics.copy()

    for idx, (train_idx, test_idx) in tqdm(list(enumerate(folds))):
        X_train, X_test = X.iloc[train_idx, :], X.iloc[test_idx, :]
        y_train, y_test = y[train_idx], y[test_idx]

        # ajuste e predicoes
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # calcula metricas
        for metric in metrics.keys():
            value = _metrics[metric]["method"](y_test, y_pred)
            _metrics[metric]["array"].append(value)

    for metric in _metrics:
        # calcula media e intervalo de confianca (95%) para cada metrica
        mean, (ci_lower, ci_upper) = summary_metric_array(metrics[metric]["array"])
        metrics[metric]["mean"] = mean
        metrics[metric]["ci_lower"] = ci_lower
        metrics[metric]["ci_upper"] = ci_upper

    score = metrics[criterion]["mean"]
    return score, _metrics


def cross_validate(
    data: pd.DataFrame,
    estimator,
    k: int,
    metrics: Dict,
    criterion: str,
    parameters: List[Dict],
):
    logger.info(f"Starting cross validation for model {estimator.model_name}...")

    # separa variaveis independentes X da variavel alvo y
    X = data.iloc[:, :-1]
    y = data["pulsar"]

    kfold = StratifiedKFold(n_splits=k)
    folds = kfold.split(X, y)

    best_params = None
    highest_score = None
    best_metrics = None

    logger.info(f"Running CV with k={k} for each hyper parameter combinarion...")
    for param_combination in parameters:
        clf = estimator.model_pipeline
        clf.set_params(**param_combination)
        score, _metrics = __run_kfolds(clf, folds, X, y, criterion, metrics)

        if not highest_score or score > highest_score:
            best_params = param_combination
            best_metrics = _metrics

    logger.info(f"Best hyper parameters founded...")
    return best_params, best_metrics
