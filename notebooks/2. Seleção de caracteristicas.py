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

# ![image.png](attachment:9ec502cf-dfeb-40b6-8bf5-3c6ddf605c12.png)

# Seleção de caracteristicas

# +
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, SelectFdr, SelectFpr, SelectFwe, SelectPercentile
from sklearn.feature_selection import f_classif, mutual_info_classif

# +
# # Dataset

DATASET_PREPROCESSED_PATH: str = "../data/preprocessed/HTRU_2_outliers_removed.csv"
# -

df_wo = pd.read_csv(DATASET_PREPROCESSED_PATH)

df_wo.head()

y = df_wo['pulsar']

X = df_wo.iloc[:, :-1]

input_features = X.columns

input_features

# # SelectKBest

k_best = SelectKBest(f_classif, k=5)

X_new = k_best.fit_transform(X, y)

df_kbest = pd.DataFrame(X_new, columns=k_best.get_feature_names_out(input_features))
df_kbest.head()

k_best_m = SelectKBest(mutual_info_classif, k=5)

X_new_m = k_best_m.fit_transform(X, y)

df_kbest_m = pd.DataFrame(X_new_m, columns=k_best_m.get_feature_names_out(input_features))
df_kbest_m.head()

# # SelectFpr

fpr = SelectFpr(f_classif, alpha=0.99)

X_fpr = fpr.fit_transform(X, y)

df_fpr = pd.DataFrame(X_fpr, columns=fpr.get_feature_names_out(input_features))
df_fpr.head()

fpr.pvalues_

# # SelectFdr

fdr = SelectFdr(f_classif, alpha=0.05)

X_fdr = fdr.fit_transform(X, y)

df_fdr = pd.DataFrame(X_fdr, columns=fdr.get_feature_names_out(input_features))
df_fdr.head()

fdr.pvalues_

# # SelectPercentile

percentile = SelectPercentile(mutual_info_classif, percentile=80)

X_percentile = percentile.fit_transform(X, y)

df_percentile = pd.DataFrame(X_percentile, columns=percentile.get_feature_names_out(input_features))
df_percentile.head()

# # SmartCorrelatedSelection

# !pip install feature-engine

from feature_engine.selection import SmartCorrelatedSelection

df_smart = pd.DataFrame(SmartCorrelatedSelection(selection_method="variance").fit_transform(X, y))
df_smart

X.corr()


