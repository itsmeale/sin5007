from typing import Dict

from feature_engine.selection import SmartCorrelatedSelection
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectPercentile, mutual_info_classif
from sklearn.preprocessing import MinMaxScaler

from src.evaluation.metrics import METRICS


class Scenario:
    name: str
    is_balanced: bool
    feature_selection: str
    has_fs: bool
    metrics: Dict
    preprocessing_steps: Pipeline
    best_params: Dict = None
    selection_criteria: str = "f1_score"


class AllCharacteristicsScenario(Scenario):
    name = "ALL CHARACTERISTICS"
    is_balanced = False
    feature_selection = "ALL CHARACTERISTICS"
    has_fs = False
    metrics = METRICS
    preprocessing_steps = [
        ("scaler", MinMaxScaler()),
    ]


class PCAScenario(Scenario):
    name = "PCA"
    is_balanced = False
    feature_selection = "PCA"
    has_fs = True
    metrics = METRICS
    preprocessing_steps = [
        ("scaler", MinMaxScaler()),
        ("pca", PCA(n_components=5)),
    ]


class MIScenario(Scenario):
    name = "Percentile"
    is_balanced = False
    feature_selection = "Select Percentile"
    has_fs = True
    metrics = METRICS
    preprocessing_steps = [
        ("scaler", MinMaxScaler()),
        ("fs", SelectPercentile(score_func=mutual_info_classif, percentile=80)),
    ]


class SmartCorrelated(Scenario):
    name = "Smart Correlated"
    is_balanced = False
    feature_selection = "Smart Correlated"
    has_fs = True
    metrics = METRICS
    preprocessing_steps = [
        ("scaler", MinMaxScaler()),
        ("fs", SmartCorrelatedSelection(selection_method="variance")),
    ]


_SCENARIOS = [
    #AllCharacteristicsScenario,
    #PCAScenario,
    #MIScenario,
    SmartCorrelated,
]

_BALANCED_SCENARIOS = list()

for _scenario in _SCENARIOS:
    _scenario = _scenario()
    _scenario.is_balanced = True
    _scenario.name = f"{_scenario.name}\n(Balanced)"
    _scenario.preprocessing_steps = [
        ("smt", SMOTETomek(random_state=0)),
        *_scenario.preprocessing_steps,
    ]
    _BALANCED_SCENARIOS.append(_scenario)


SCENARIOS = [
    # *[_scenario() for _scenario in _SCENARIOS],
    *_BALANCED_SCENARIOS,
]
