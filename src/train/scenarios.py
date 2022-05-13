from typing import Dict

from imblearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectPercentile, mutual_info_classif
from sklearn.preprocessing import MinMaxScaler

from src.evaluation.metrics import METRICS


class Scenario:
    name: str
    is_balanced: bool
    feature_selection: str
    metrics: Dict
    preprocessing_steps: Pipeline
    best_params: Dict = None
    selection_criteria: str = "f1_score"


class PCAScenario(Scenario):
    name = "PCA"
    is_balanced = False
    feature_selection = None
    metrics = METRICS
    preprocessing_steps = [
        ("scaler", MinMaxScaler()),
        ("pca", PCA(n_components=5)),
    ]


class MIScenario(Scenario):
    name = "MI"
    is_balanced = False
    feature_selection = "Mutual Information"
    metrics = METRICS
    preprocessing_steps = [
        ("fs", SelectPercentile(score_func=mutual_info_classif, percentile=80)),
    ]


class NoPrepScenario(Scenario):
    name = "NO PREPROCESSING"
    is_balanced = False
    feature_selection = None
    metrics = METRICS
    preprocessing_steps = []


SCENARIOS = [
    NoPrepScenario(),
    PCAScenario(),
    MIScenario(),
]
