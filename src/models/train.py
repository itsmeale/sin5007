from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from src.features.preprocessing import PREPROCESSING_PIPELINES

MODELS = {
    "tree": {
        "classifier": DecisionTreeClassifier(),
        "param_grid": dict(),  # TODO: sera utilizado no futuro para parameter tunning
    },
    "forest": {
        "classifier": RandomForestClassifier(),
        "param_grid": dict(),
    },
}


def generate_classifiers():
    """
    Gera um dicionario com as permutacoes de classificadores e pipelines de preprocessamento,
    o dicionario tem a seguinte estrutura:
    {
        "model_name_and_preprocess_steps": {
            "classifier": Modelo(),  # Modelo instanciado.
            "param_grid": dict(),  # dicionario que pode ser utilizado para tunagem de hiperparametros.
        }
    }
    """
    classifiers = dict()
    for classifier_name, classifier_meta in MODELS.items():
        for (
            preprocess_pipe_name,
            preprocess_pipeline,
        ) in PREPROCESSING_PIPELINES.items():
            pipe_name = f"{classifier_name}_{preprocess_pipe_name}"
            classifiers[pipe_name] = {
                "classifier": Pipeline(
                    steps=[
                        *preprocess_pipeline.steps,
                        ("classifier", classifier_meta["classifier"]),
                    ]
                ),
                "param_grid": classifier_meta["param_grid"],
            }
    return classifiers


classifiers = generate_classifiers()
