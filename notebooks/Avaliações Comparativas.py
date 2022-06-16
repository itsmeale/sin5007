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

# # Avaliações comparativas

# +
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from yellowbrick.classifier import PrecisionRecallCurve, ROCAUC

import matplotlib.pyplot as plt
# -

df = pd.read_csv("../data/preprocessed/HTRU_2_outliers_removed.csv")

X, y = df.iloc[:, :-1], df["pulsar"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

metrics = pd.read_csv("../data/results/metrics.csv")

metrics[metrics.scenario_name=="ALL CHARACTERISTICS"]

# ## Models pipelines

gb_pipe = Pipeline(steps=[
    ("scaler", MinMaxScaler()),
    ("clf", GaussianNB(var_smoothing=0.01))
])

svc_pipe = Pipeline(steps=[
    ("scaler", MinMaxScaler()),
    ("clf", SVC(kernel="poly", C=100, degree=3, gamma=1, probability=True))
])

svc_pipe.fit(X_train, y_train)
gb_pipe.fit(X_train, y_train)

y_proba_svc = svc_pipe.predict_proba(X_test)[:, -1]
y_pred_svc = svc_pipe.predict(X_test)

y_proba_gb = gb_pipe.predict_proba(X_test)[:, -1]
y_pred_gb = gb_pipe.predict(X_test)

# ## PR Curves

from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
from matplotlib.ticker import MultipleLocator

plt.style.use('default')
plt.rcParams['xtick.major.size'] = 5.0
plt.rcParams['xtick.minor.size'] = 3.0
plt.rcParams['ytick.major.size'] = 5.0
plt.rcParams['ytick.minor.size'] = 3.0
plt.rcParams['axes.linewidth'] = 2.0
plt.rcParams['xtick.direction'] = "in"
plt.rcParams['ytick.direction'] = "in"

# +
fig, ax = plt.subplots(dpi=150, figsize=(8, 6))
ax.yaxis.set_minor_locator(MultipleLocator(.1))
ax.xaxis.set_minor_locator(MultipleLocator(.1))

PrecisionRecallDisplay.from_predictions(y_test, y_proba_svc, ax=ax, name="SVC", marker='o', markersize=2, linewidth=1, color="#6B62E3")
PrecisionRecallDisplay.from_predictions(y_test, y_proba_gb, ax=ax, name="GaussianNB", marker='s', markersize=2, linewidth=1, color="#E34334")

ax.set_title("Precision-Recall")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")

plt.grid(False)
plt.show()
# -

# ## ROC Curves

# +
fig, ax = plt.subplots(dpi=150, figsize=(8, 6))
ax.yaxis.set_minor_locator(MultipleLocator(.1))
ax.xaxis.set_minor_locator(MultipleLocator(.1))

RocCurveDisplay.from_predictions(y_test, y_pred_svc, ax=ax, name="SVC", marker='o', markersize=2, linewidth=2, color="#6B62E3")
RocCurveDisplay.from_predictions(y_test, y_pred_gb, ax=ax, name="GaussianNB", marker='s', markersize=2, linewidth=2, color="#E34334")

ax.set_title("ROC Curve")

plt.grid(False)
plt.show()
# -


