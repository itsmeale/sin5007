from typing import Dict

from imblearn.pipeline import Pipeline
from feature_engine.selection import SmartCorrelatedSelection
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


class AllCharacteristicsScenario(Scenario):
    name = "ALL CHARACTERISTICS"
    is_balanced = False
    feature_selection = "ALL CHARACTERISTICS"
    metrics = METRICS
    preprocessing_steps = []

class PCAScenario(Scenario):
    name = "PCA"
    is_balanced = False
    feature_selection = "PCA"
    metrics = METRICS
    preprocessing_steps = [
        ("scaler", MinMaxScaler()),
        ("pca", PCA(n_components=5)),
    ]

class MIScenario(Scenario):
    name = "Percentile"
    is_balanced = False
    feature_selection = "Select Percentile"
    metrics = METRICS
    preprocessing_steps = [
        ("fs", SelectPercentile(score_func=mutual_info_classif, percentile=80)),
    ]

class SmartCorrelated(Scenario):
    name = "Smart Correlated"
    is_balanced = False
    feature_selection = "Smart Correlated"
    metrics = METRICS
    preprocessing_steps = [
        ("fs", SmartCorrelatedSelection(selection_method="variance")),
    ]    



SCENARIOS = [
    AllCharacteristicsScenario(),
    PCAScenario(),
    MIScenario(),
    SmartCorrelated(),
]
