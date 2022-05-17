from typing import Dict, List

from sklearn.base import ClassifierMixin
from sklearn.naive_bayes import GaussianNB


class Model:
    name: str
    classifier: ClassifierMixin
    param_grid: List[Dict]


class NaiveBayes(Model):
    name = "Naive Bayes"
    classifier = GaussianNB()
    param_grid = [{"clf__var_smoothing": [1e-9, 1e-5, 1e-3, 1e-2, 1e-1]}]


MODELS = [
    NaiveBayes(),
]
