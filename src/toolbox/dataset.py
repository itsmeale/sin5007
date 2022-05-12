import numpy as np


def __get_positives_and_negatives(y):
    """Calcula proporcao de positivos e negativos com base no vetor alvo y"""
    t = len(y)
    p = len(np.where(y == 1)[0])
    n = t - p
    return t, p, n


def print_dataset_balance(y):
    t, positive, negative = __get_positives_and_negatives(y)

    print(
        f"{positive} positivas e {negative} negativas "
        f"({positive/t:.2%} x {negative/t:.2%})"
    )
