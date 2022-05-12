import pandas as pd
from sklearn.naive_bayes import GaussianNB

from src.evaluation.metrics import METRICS
from src.models.estimators import Estimator

NAIVE_BAYES = Estimator(
    model_name="naive_bayes",
    is_balanced=False,
    feature_selection=None,
    param_grid=[{"var_smoothing": [1e-9, 1e-5]}],
    metrics=METRICS,
    model_pipeline=GaussianNB,
)


MODELS = [
    NAIVE_BAYES,
]


if __name__ == "__main__":
    df = pd.read_csv("data/preprocessed/HTRU_2_outliers_removed.csv")

    for model in MODELS:
        model.evaluate(df)
        model.save_metrics()
