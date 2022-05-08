import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def __get_positives_and_negatives(y):
    t = len(y)
    p = len(np.where(y == 1)[0])
    n = t - p
    return t, p, n    


def cross_validate(df: pd.DataFrame, k: int):
    X = df.iloc[:, :-1]
    y = df["pulsar"]
    
    t, positive, negative = __get_positives_and_negatives(y)

    print(
        f"{positive} positivas e {negative} negativas "
        f"({positive/t:.2%} x {negative/t:.2%})"
    )

    kfold = StratifiedKFold(n_splits=k)

    for idx, (train_idx, test_idx) in enumerate(kfold.split(X, y)):        
        X_train, X_test = X.iloc[train_idx, :], X.iloc[test_idx, :]
        y_train, y_test = y[train_idx], y[test_idx]

        fold_t, fold_p, fold_n = __get_positives_and_negatives(y_train)

        print(
            f"Fold {idx+1}: Pos: {fold_p}, Neg: {fold_n}, Total: {fold_t}, "
            f"Proporção: {fold_p/fold_t:.2%};{fold_n/fold_t:.2%}"
        )


if __name__ == "__main__":
    df = pd.read_csv("data/preprocessed/HTRU_2_outliers_removed.csv")
    cross_validate(df, k=10)
