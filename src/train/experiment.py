from dataclasses import dataclass
from itertools import product
from pathlib import Path

import pandas as pd
from imblearn.pipeline import Pipeline
from loguru import logger
from sklearn import metrics
from slugify import slugify

from src.evaluation.crossval import cross_validate
from src.train.models import Model
from src.train.scenarios import Scenario

K_FOLDS = 10
MODEL_SELECTION_CRITERIA = "f1_score"


@dataclass
class Experiment:

    scenario: Scenario
    model: Model

    def run(self, X, y):
        """Runs cross validation with k folds and returns best hyper parameters"""
        logger.info(
            f"Running experiment for {self.model.name} with {self.scenario.name}"
        )

        best_params, best_metrics = cross_validate(
            X=X,
            y=y,
            scenario=self.scenario,
            model=self.model,
            k=K_FOLDS,
            metrics=self.scenario.metrics,
            criterion=self.scenario.selection_criteria,  # to choose the best model
            parameters=self.get_hyperparameters_combinations(),
            experiment=self,
        )

        self.scenario.metrics.update(best_metrics)
        self.scenario.best_params = best_params
        return self

    def get_hyperparameters_combinations(self):
        """Creates the combinations of hyperparameters that are defined on the param grid"""
        parameter_combinations = [
            dict(zip(grid.keys(), combination))
            for grid in self.model.param_grid
            for combination in list(product(*grid.values()))
        ]

        return parameter_combinations

    def __create_parent_results_dir(self, metrics_folder):
        path = Path(f"{metrics_folder}/{slugify(self.model.name)}")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_metrics(self, metrics_folder: str):
        """Serialize metrics and estimator parameters to csv"""
        logger.info(f"Saving metrics for model {self.model.name}...")

        serialized_metrics_df = pd.DataFrame(
            {
                "model_name": [self.model.name],
                "scenario_name": [self.scenario.name],
                "is_balanced": [self.scenario.is_balanced],
                "feature_selection": [self.scenario.feature_selection],
                "param_grid": [self.model.param_grid],
                "best_params": [self.scenario.best_params],
                # accuracy metrics
                "accuracy": [self.scenario.metrics["accuracy"]["mean"]],
                "accuracy_ci_lower": [self.scenario.metrics["accuracy"]["ci_lower"]],
                "accuracy_ci_upper": [self.scenario.metrics["accuracy"]["ci_upper"]],
                # precision metrics
                "precision": [self.scenario.metrics["precision"]["mean"]],
                "precision_ci_lower": [self.scenario.metrics["precision"]["ci_lower"]],
                "precision_ci_upper": [self.scenario.metrics["precision"]["ci_upper"]],
                # recall metrics
                "recall": [self.scenario.metrics["recall"]["mean"]],
                "recall_ci_lower": [self.scenario.metrics["recall"]["ci_lower"]],
                "recall_ci_upper": [self.scenario.metrics["recall"]["ci_upper"]],
                # f1 metrics
                "f1_score": [self.scenario.metrics["f1_score"]["mean"]],
                "f1_score_ci_lower": [self.scenario.metrics["f1_score"]["ci_lower"]],
                "f1_score_ci_upper": [self.scenario.metrics["f1_score"]["ci_upper"]],
            }
        )

        file_name = slugify(f"{self.model.name} {self.scenario.name}")
        output_folder = self.__create_parent_results_dir(metrics_folder=metrics_folder)

        serialized_metrics_df.to_csv(f"{output_folder}/{file_name}.csv", index=False)
        return self


if __name__ == "__main__":
    from src.train.models import SVM, NaiveBayes
    from src.train.scenarios import PCAScenario

    df = pd.read_csv("data/preprocessed/HTRU_2_outliers_removed.csv")
    X = df.iloc[:, :-1]
    y = df["pulsar"]

    experiment = Experiment(scenario=PCAScenario(), model=SVM())
    experiment.run(X, y).save_metrics(metrics_folder="data/results")
