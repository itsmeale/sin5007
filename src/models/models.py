import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectPercentile, mutual_info_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler

from src.dataviz.plots import make_bar_chart_comparision
from src.evaluation.metrics import METRICS, aggregate_metrics
from src.models.estimators import Estimator

NAIVE_BAYES_PCA = Estimator(
    model_name="naive_bayes",
    scenario_name="PCA",
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
    model_name="naive_bayes",
    scenario_name="ENTROPY",
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

NAIVE_BAYES = Estimator(
    model_name="naive_bayes",
    scenario_name="NO PREPROCESSING",
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
    model_name="naive_bayes",
    scenario_name="NO PREPROCESSING",
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
    NAIVE_BAYES_PCA,
    NAIVE_BAYES_FS,
]


if __name__ == "__main__":
    df = pd.read_csv("data/preprocessed/HTRU_2_outliers_removed.csv")
    metrics_folder = "data/results"
    metrics_file = "data/results/metrics.csv"

    for model in MODELS:
        model.evaluate(df)
        model.save_metrics(metrics_folder)

    aggregate_metrics(metrics_folder=metrics_folder, save_to=metrics_file)
    metrics_df = pd.read_csv(metrics_file)

    make_bar_chart_comparision(metrics_df)
