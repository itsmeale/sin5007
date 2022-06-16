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

# +
X = df.iloc[:, :-1].values
y = df["pulsar"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.2, random_state=0)
# -

# ---

# ## TabNet

from pytorch_tabnet.tab_model import TabNetClassifier

clf = TabNetClassifier()

clf.fit(X_train, y_train, eval_set=[(X_val, y_val)])

y_pred= clf.predict(X_test)

from sklearn.metrics import f1_score

f1_score(y_test, y_pred)


