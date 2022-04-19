import pandas as pd

from sklearn.model_selection import train_test_split


def main():
    """Separa o conjunto de dados original em um conjunto de desenvolvimento e um conjunto de testes.

    Idealmente, todo preprocessamento deverá ser feito apenas sobre o conjunto de desenvolvimento para evitar
    problemas com data leakage. A proporção de separação é de 70% das amostras para desenvolvimento e 30% para testes.
    """
    complete_df = pd.read_csv("data/raw/HTRU_2.csv")
    features = complete_df.columns.tolist()[:-1]
    target = complete_df.columns.tolist()[-1]

    X_train, X_test, y_train, y_test = train_test_split(
        complete_df[features],
        complete_df[target],
        test_size=0.3,
        random_state=0,
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_df.to_csv("data/raw/dev_set.csv", index=False)
    test_df.to_csv("data/raw/test_set.csv", index=False)


if __name__ == "__main__":
    main()
