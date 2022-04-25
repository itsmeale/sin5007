import pandas as pd
from sklearn.neighbors import LocalOutlierFactor


def remove_outliers(df):
    lof = LocalOutlierFactor()
    outlier_indexes = lof.fit_predict(df)
    return df[outlier_indexes == 1]


def main():
    df = pd.read_csv("data/raw/HTRU_2.csv")
    df = remove_outliers(df)
    df.to_csv("data/preprocessed/HTRU_2_outliers_removed.csv", index=False)


if __name__ == "__main__":
    main()
