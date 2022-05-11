from typing import Dict

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

METRICS: Dict = {
    "recall": {
        "method": recall_score,
        "array": list(),
        "mean": None,
        "ci_lower": None,
        "ci_upper": None,
    },
    "precision": {
        "method": precision_score,
        "array": list(),
        "mean": None,
        "ci_lower": None,
        "ci_upper": None,
    },
    "accuracy": {
        "method": accuracy_score,
        "array": list(),
        "mean": None,
        "ci_lower": None,
        "ci_upper": None,
    },
    "f1_score": {
        "method": f1_score,
        "array": list(),
        "mean": None,
        "ci_lower": None,
        "ci_upper": None,
    },
}
