from copy import deepcopy
from typing import Dict, List

import numpy as np
import statsmodels.stats.api as stats
from imblearn.pipeline import Pipeline
from loguru import logger
from sklearn.model_selection import StratifiedKFold


def summary_metric_array(metric_array: List):
    """Calcula media e intervalo de confianca par ao vetor metric_array"""
    mean = np.mean(metric_array)
    ci = stats.DescrStatsW(metric_array).tconfint_mean()
    return mean, ci


def __run_kfolds(scenario, model, params, k, X, y, criterion, metrics):
    _metrics = deepcopy(metrics)

    kfold = StratifiedKFold(n_splits=k)
    folds = kfold.split(X, y)

    max_features_param = "clf__max_features"
    if scenario.has_fs and max_features_param in params.keys():
        params[max_features_param] = min(5, params[max_features_param])

    for (train_idx, test_idx) in folds:
        X_train, X_test = X.iloc[train_idx, :], X.iloc[test_idx, :]
        y_train, y_test = y[train_idx], y[test_idx]

        preprocessor = Pipeline(steps=scenario.preprocessing_steps)
        X_train = preprocessor.fit_transform(X_train)

        clf = Pipeline(
            steps=[
                ("clf", model.classifier(**model.fixed_params)),
            ]
        )

        clf.set_params(**params)

        # ajuste e predicoes
        clf.fit(
            X_train,
            y_train,
            **model.fit_params,
        )
        y_pred = clf.predict(preprocessor.transform(X_test))

        # calcula metricas
        for metric in metrics.keys():
            value = _metrics[metric]["method"](y_test, y_pred)
            _metrics[metric]["array"].append(value)

    for metric in _metrics:
        # calcula media e intervalo de confianca (95%) para cada metrica
        mean, (ci_lower, ci_upper) = summary_metric_array(_metrics[metric]["array"])
        _metrics[metric]["mean"] = mean
        _metrics[metric]["ci_lower"] = ci_lower
        _metrics[metric]["ci_upper"] = ci_upper

    score = _metrics[criterion]["mean"]
    logger.info(f"{model.name}: {score}")
    return score, _metrics


def cross_validate(
    X,
    y,
    scenario,
    model,
    k: int,
    metrics: Dict,
    criterion: str,
    parameters: List[Dict],
):
    logger.info(f"Starting cross validation...")

    best_params = None
    highest_score = None
    best_metrics = None

    logger.info(f"Running CV with k={k} for each hyper parameter combinarion...")
    for _, param_combination in enumerate(parameters):
        score, _metrics = __run_kfolds(
            scenario, model, param_combination, k, X, y, criterion, metrics
        )

        if not highest_score or score > highest_score:
            highest_score = score
            best_params = param_combination
            best_metrics = _metrics

    logger.info(f"Best hyper parameters founded...")
    return best_params, best_metrics
