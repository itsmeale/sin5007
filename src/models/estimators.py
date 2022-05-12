from dataclasses import dataclass
from itertools import product
from pprint import pprint
from typing import Dict, List

import pandas as pd
from loguru import logger
from sklearn.pipeline import Pipeline

from src.evaluation.crossval import cross_validate
from src.evaluation.metrics import METRICS

K_FOLDS = 10
MODEL_SELECTION_CRITERIA = "f1_score"


@dataclass
class Estimator:

    model_name: str
    is_balanced: bool
    feature_selection: str
    param_grid: List[Dict]
    metrics: Dict
    model_pipeline: Pipeline
    best_params: Dict = None

    def evaluate(self, df: pd.DataFrame):
        """Runs cross validation with k folds and returns best hyper parameters"""
        best_params, best_metrics = cross_validate(
            data=df,
            estimator=self,
            k=K_FOLDS,
            metrics=self.metrics,
            criterion=MODEL_SELECTION_CRITERIA,  # to choose the best model
            parameters=self.get_hyperparameters_combinations(),
        )
        self.metrics.update(best_metrics)
        self.best_params = best_params

    def get_hyperparameters_combinations(self):
        """Creates the combinations of hyperparameters that are defined on the param grid"""
        parameter_combinations = [
            dict(zip(grid.keys(), combination))
            for grid in self.param_grid
            for combination in list(product(*grid.values()))
        ]

        return parameter_combinations

    def save_metrics(self, metrics_folder: str):
        """Serialize metrics and estimator parameters to csv"""
        logger.info(f"Saving metrics for model {self.model_name}...")

        serialized_metrics_df = pd.DataFrame(
            {
                "model_name": [self.model_name],
                "is_balanced": [self.is_balanced],
                "feature_selection": [self.feature_selection],
                "param_grid": [self.param_grid],
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

        serialized_metrics_df.to_csv(
            f"{metrics_folder}/{self.model_name}.csv", index=False
        )
