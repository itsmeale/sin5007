from dataclasses import dataclass
from typing import Dict

import pandas as pd
from sklearn.pipeline import Pipeline

from src.evaluation.crossval import cross_validate
from src.evaluation.metrics import METRICS


@dataclass
class Estimator:

    model_name: str
    is_balanced: bool
    feature_selection: str
    params: Dict
    best_params: Dict
    metrics: Dict
    model_pipeline: Pipeline

    def evaluate(self, df: pd.DataFrame):
        """Run cross validation for the model_pipeline"""
        cross_validate(
            data=df, estimator=self.model_pipeline, k=10, metrics=self.metrics
        )

    def save_metrics(self):
        """Serialize metrics and estimator parameters to csv"""
        serialized_metrics_df = pd.DataFrame(
            {
                "model_name": [self.model_name],
                "balance_dataset": [self.balance_dataset],
                "feature_selection": [self.feature_selection],
                "params": [self.params],
                "best_params": [self.best_params],
                # accuracy metrics
                "accuracy": [self.metrics["accuracy"]["mean"]],
                "accuracy_ci_lower": [self.metrics["accuracy"]["ci_lower"]],
                "accuracy_ci_upper": [self.metrics["accuracy"]["ci_upper"]],
                # precision metrics
                "precision": [self.metrics["precision"]["mean"]],
                "precision_ci_lower": [self.metrics["precision"]["ci_lower"]],
                "precision_ci_upper": [self.metrics["precision"]["ci_upper"]],
                # recall metrics
                "recall": [self.metrics["recall"]["mean"]],
                "recall_ci_lower": [self.metrics["recall"]["ci_lower"]],
                "recall_ci_upper": [self.metrics["recall"]["ci_upper"]],
                # f1 metrics
                "f1_score": [self.metrics["f1_score"]["mean"]],
                "f1_score_ci_lower": [self.metrics["f1_score"]["ci_lower"]],
                "f1_score_ci_upper": [self.metrics["f1_score"]["ci_upper"]],
            }
        )

        serialized_metrics_df.to_csv(f"data/results/{self.model_name}.csv", index=False)


if __name__ == "__main__":
    estimator = Estimator(
        model_name="meu_modelo_1",
        balance_dataset=1,
        feature_selection=None,
        params={"MAP": 1},
        metrics=METRICS,
        model_pipeline=None,
    )
    estimator.save_metrics()
