from typing import Dict, List

from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


class Model:
    name: str
    classifier: ClassifierMixin
    param_grid: List[Dict]


class NaiveBayes(Model):
    name = "Naive Bayes"
    classifier = GaussianNB()
    param_grid = [{"clf__var_smoothing": [1e-9, 1e-5]}]


class RandomForest(Model):
    name = "Random Forest"
    classifier = RandomForestClassifier()
    param_grid = [{"clf__n_estimators": [100, 200, 1000]}]


MODELS = [
    NaiveBayes(),
    RandomForest(),
]
