from typing import Dict, List

from sklearn.base import ClassifierMixin
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


class Model:
    name: str
    classifier: ClassifierMixin
    param_grid: List[Dict]


class NaiveBayes(Model):
    name = "Naive Bayes"
    classifier = GaussianNB()
    param_grid = [{"clf__var_smoothing": [1e-9, 1e-5, 1e-3, 1e-2, 1e-1]}]


class SVM(Model):
    name = "SVM"
    classifier = SVC(random_state=0)
    param_grid: List[Dict] = [
        {
            "clf__kernel": ["linear"],
            "clf__C": [1, 0.5],
            "clf__class_weight": [None, "balanced"],
        },
        {
            "clf__kernel": ["poly"],
            "clf__C": [1, 0.5],
            "clf__degree": [1, 2, 3, 4],
            "clf__gamma": [1, 1.5, 1e-1, 1e-2],
            "clf__class_weight": [None, "balanced"],
        },
        {
            "clf__kernel": ["rbf"],
            "clf__C": [1, 0.5],
            "clf__gamma": [1, 1.5, 1e-1, 1e-2],
            "clf__class_weight": [None, "balanced"],
        },
    ]


MODELS = [
    SVM(),
]
