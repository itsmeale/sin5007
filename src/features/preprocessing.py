import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE, VarianceThreshold


random_state = 0


PIPE_SCALING_BALANCING = Pipeline(
    steps=[
        ("scaler", MinMaxScaler()),
        ("smote", SMOTE(random_state=random_state)),
    ]
)

PIPE_SCALING_BALANCING_PCA = Pipeline(
    steps=[
        *PIPE_SCALING_BALANCING.steps,
        ("pca", PCA()),
    ]
)

PIPE_SCALING_BALANCING_VARTHRESH = Pipeline(
    steps=[
        *PIPE_SCALING_BALANCING.steps,
        ("var_threshold", VarianceThreshold()),
    ]
)

PIPE_SCALING = Pipeline(
    steps=[
        ("scaler", MinMaxScaler()),
    ]
)

PIPE_SCALING_PCA = Pipeline(
    steps=[
        ("scaler", MinMaxScaler()),
        ("pca", PCA()),
    ]
)

PIPE_SCALING_VARTHRESH = Pipeline(
    steps=[
        ("scaler", MinMaxScaler()),
        ("var_threshold", VarianceThreshold()),
    ]
)

PREPROCESSING_PIPELINES = {
    # balanced datasets
    "scaling_and_balancing": PIPE_SCALING_BALANCING,
    "scaling_balancing_pca": PIPE_SCALING_BALANCING_PCA,
    "scaling_balancing_vartresh": PIPE_SCALING_BALANCING_VARTHRESH,
    # unbalanced datatset
    "only_scaling": PIPE_SCALING,
    "scaling_pca": PIPE_SCALING_PCA,
    "scaling_vartresh": PIPE_SCALING_VARTHRESH,
}
