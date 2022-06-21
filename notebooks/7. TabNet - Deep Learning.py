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

# # Redes Neurais Artificiais

import pandas as pd

df = pd.read_csv("../data/preprocessed/HTRU_2_outliers_removed.csv")

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# +
X = df.iloc[:, :-1].values
y = df["pulsar"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train, X_val, y_train, X_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
# -

# ---

# ## TabNet

from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import StratifiedKFold

# Metric

# +
import numpy as np
from pytorch_tabnet.metrics import Metric
from sklearn.metrics import f1_score

class F1(Metric):
    def __init__(self):
        self._name = "F1"
        self._maximize = True
    
    def __call__(self, y_true, y_pred):
        y_pred = np.where(y_pred[:, 1] >= 0.5, 1, 0)
        return f1_score(y_true, y_pred)


# -

def tabnet(trial):
    skf = StratifiedKFold(n_splits=3)
    scores = []

    n_d = trial.suggest_int("n_d", 8, 20)
    n_a = trial.suggest_int("n_a", 8, 20)
    n_steps = trial.suggest_int("n_steps", 3, 10)
    gamma = trial.suggest_float("gamma", 1, 2)
    lr = trial.suggest_float("lr", 1e-3, 1e0)

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.15, stratify=y_train)

        clf = TabNetClassifier(
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            optimizer_params={
                "lr": lr
            }
        )
        clf.fit(
            X_train,
            y_train,
            eval_metric=[F1],
            eval_set=[(X_val, y_val)],
            max_epochs=200,
            batch_size=1800,
            patience=50
        )
        score = f1_score(y_test, clf.predict(X_test))
        scores.append(score)

        return np.mean(scores)


import optuna

study = optuna.create_study(direction="maximize")
study.optimize(tabnet, n_trials=100)


