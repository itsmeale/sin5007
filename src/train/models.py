from typing import Dict, List

from sklearn.base import ClassifierMixin
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


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
    classifier = SVC(random_state=0, max_iter=50000)
    param_grid: List[Dict] = [
        {
            "clf__kernel": ["linear"],
            "clf__C": [1e-1, 1, 1e1, 1e2],
        },
        {
            "clf__kernel": ["sigmoid"],
            "clf__C": [1e-1, 1, 1e1, 1e2],
            "clf__gamma": [1e-2, 1e-1, 1, 1e2, 1e3],
        },
        {
            "clf__kernel": ["rbf"],
            "clf__C": [1e-1, 1, 1e1, 1e2],
            "clf__gamma": [1e-2, 1e-1, 1, 1e2, 1e3],
        },
        {
            "clf__kernel": ["poly"],
            "clf__C": [1e-1, 1, 1e1, 1e2],
            "clf__degree": [1, 2, 3],
            "clf__gamma": [1e-2, 1e-1, 1, 1e2, 1e3],
        },
    ]


class MLP(Model):
    name = "MLP"
    classifier = MLPClassifier(random_state=0, max_iter=200, learning_rate="constant", early_stopping=True)
    param_grid: List[Dict] = [
        {
            "clf__hidden_layer_sizes": [(10,), (3,),],
            "clf__activation": ["identity", "logistic", "tanh", "relu"],
            "clf__learning_rate_init": [1e-3, 1e-2, 1e-1],
            "clf__solver": ["adam"],
        }
    ]


MODELS = [
    MLP(),
]
