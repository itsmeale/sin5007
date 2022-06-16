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

# # Testando otimização de hiperparâmetros com Optuna
# ---
#
# Hipótese: utilizar optuna para tunar os hiperparâmetros pode fornecer um modelo com maior f1-score.

import optuna
import pandas as pd

df = pd.read_csv("../data/preprocessed/HTRU_2_outliers_removed.csv")
y = df["pulsar"].values
X = df.drop("pulsar", axis=1).values

# ---

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score


# melhores parâmetros encontrados via grid search

def cross_validation(estimator, X, y):
    skf = StratifiedKFold(n_splits=10)
    scores = list()

    for train_index, test_index in skf.split(X, y):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        scores.append(f1_score(y_test, y_pred))
    
    return scores


pipe = make_pipeline(MinMaxScaler(), SVC(kernel="poly", C=100, degree=3, gamma=1))  # 0.8840
svc_gs_scores = cross_validation(pipe, X, y)
print(np.mean(svc_gs_scores))


# otimização com optuna

def svc_model(trial):
    C = trial.suggest_float("C", 90, 110)
    gamma = trial.suggest_float("gamma", 0.1, 2)

    pipe = make_pipeline(MinMaxScaler(), SVC(kernel="poly", C=C, degree=3, gamma=gamma))
    scores = cross_validation(pipe, X, y)

    return np.mean(scores)


study = optuna.create_study(direction="maximize")
study.optimize(svc_model, n_trials=20)

# Validação cruzada com melhores parâmetros encontrados pelo optuna

pipe_opt = make_pipeline(MinMaxScaler(), SVC(kernel="poly", degree=3, **study.best_params))

svc_opt_scores = cross_validation(pipe_opt, X, y)
np.mean(svc_opt_scores)

pipe_opt

# ---

# ## Teste de hipótese

from scipy.stats import ttest_rel

# +
p = ttest_rel(svc_opt_scores, svc_gs_scores).pvalue

if p <= .05:
    print("A melhoria é significativa.")
else:
    print("Melhoria não é significativa.")
# -

import matplotlib.pyplot as plt

# +
fig, ax = plt.subplots(dpi=150)

ax.hist(svc_opt_scores, label="Optuna", bins=np.arange(.8, 1, 0.025), edgecolor='k', alpha=.4)
ax.hist(svc_gs_scores, label="Grid Search", bins=np.arange(.8, 1, .025), edgecolor='k', alpha=.4)

plt.legend()
plt.show()
# -


