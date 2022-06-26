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

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# modelos
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
# -

df = pd.read_csv("../data/preprocessed/HTRU_2_outliers_removed.csv")

X, y = df.iloc[:, :-1], df["pulsar"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

# ## Models pipelines

# +
nb = make_pipeline(MinMaxScaler(), GaussianNB(var_smoothing=.01))
svm = make_pipeline(MinMaxScaler(), SVC(kernel="poly", C=100, gamma=1, degree=3, random_state=0))
mlp = make_pipeline(
    MinMaxScaler(),
    MLPClassifier(
        hidden_layer_sizes=(10,),
        activation="tanh",
        learning_rate_init=0.1,
        solver="adam",
        learning_rate="constant",
        random_state=0,
    )
)
rf = make_pipeline(
    MinMaxScaler(),
    RandomForestClassifier(
        n_estimators=1000,
        criterion="entropy",
        max_depth=200,
        max_features=5
    )
)

models = [nb, svm, mlp, rf]
# -

for model in models:
    model.fit(X_train, y_train)

from sklearn.metrics import classification_report

for model in models:
    y_pred = model.predict(X_test)
    print(model)
    print(classification_report(y_test, y_pred))

from sklearn.ensemble import VotingClassifier

# +
clf = VotingClassifier(estimators=[
    ("svc", SVC(kernel="poly", C=100, gamma=1, degree=3, random_state=0)),
    ("mlp", MLPClassifier(
        hidden_layer_sizes=(10,),
        activation="tanh",
        learning_rate_init=0.1,
        solver="adam",
        learning_rate="constant",
        random_state=0,
    )),
    ("rf", RandomForestClassifier(
        n_estimators=1000,
        criterion="entropy",
        max_depth=200,
        max_features=5
    ))
], n_jobs=-1)

pipe = make_pipeline(MinMaxScaler(), clf)
# -

clf.fit(X_train ,y_train)

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))

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

PrecisionRecallDisplay.from_predictions(y_test, y_proba_svc, ax=ax, name="SVC (GS)", marker='o', markersize=2, linewidth=1, color="#6B62E3")
PrecisionRecallDisplay.from_predictions(y_test, y_proba_svc_opt, ax=ax, name="SVC (OPT)", marker='o', markersize=2, linewidth=1, color="#000")
PrecisionRecallDisplay.from_predictions(y_test, y_proba_gb, ax=ax, name="GaussianNB", marker='s', markersize=2, linewidth=1, color="#E34334")

ax.set_title("Precision-Recall")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")

plt.grid(False)
plt.show()
# -


