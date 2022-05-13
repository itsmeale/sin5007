from sklearn.base import ClassifierMixin
from typing import List, Dict
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


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
]
