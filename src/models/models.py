import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectPercentile, mutual_info_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler

from src.evaluation.metrics import METRICS
from src.models.estimators import Estimator

NAIVE_BAYES_PCA = Estimator(
    model_name="naive_bayes_pca",
    is_balanced=False,
    feature_selection="PCA",
    param_grid=[{"clf__var_smoothing": [1e-9, 1e-5]}],
    metrics=METRICS,
    model_pipeline=Pipeline(
        steps=[
            ("scaler", MinMaxScaler()),
            ("pca", PCA(n_components=5)),
            ("clf", GaussianNB()),
        ]
    ),
)

NAIVE_BAYES_FS = Estimator(
    model_name="naive_bayes_mi",
    is_balanced=False,
    feature_selection="MI",
    param_grid=[{"clf__var_smoothing": [1e-9, 1e-5]}],
    metrics=METRICS,
    model_pipeline=Pipeline(
        steps=[
            ("fs", SelectPercentile(score_func=mutual_info_classif, percentile=80)),
            ("clf", GaussianNB()),
        ]
    ),
)

NAIVE_BAYES_FS_MIN_MAX = Estimator(
    model_name="naive_bayes_fs_min_max",
    is_balanced=False,
    feature_selection="MI",
    param_grid=[{"clf__var_smoothing": [1e-9, 1e-5]}],
    metrics=METRICS,
    model_pipeline=Pipeline(
        steps=[
            ("scaler", MinMaxScaler()),
            ("fs", SelectPercentile(score_func=mutual_info_classif, percentile=80)),
            ("clf", GaussianNB()),
        ]
    ),
)

NAIVE_BAYES = Estimator(
    model_name="naive_bayes",
    is_balanced=False,
    feature_selection=None,
    param_grid=[{"clf__var_smoothing": [1e-9, 1e-5]}],
    metrics=METRICS,
    model_pipeline=Pipeline(
        steps=[
            ("clf", GaussianNB()),
        ]
    ),
)

NAIVE_BAYES_SMOTE = Estimator(
    model_name="naive_bayes_smote",
    is_balanced=True,
    feature_selection=None,
    param_grid=[{"clf__var_smoothing": [1e-9, 1e-5]}],
    metrics=METRICS,
    model_pipeline=Pipeline(
        steps=[
            ("smote", SMOTE()),
            ("clf", GaussianNB()),
        ]
    ),
)


MODELS = [
    NAIVE_BAYES,
    NAIVE_BAYES_SMOTE,
    NAIVE_BAYES_PCA,
    NAIVE_BAYES_FS,
    NAIVE_BAYES_FS_MIN_MAX,
]


if __name__ == "__main__":
    df = pd.read_csv("data/preprocessed/HTRU_2_outliers_removed.csv")

    for model in MODELS:
        model.evaluate(df)
        model.save_metrics()
