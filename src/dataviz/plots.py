from typing import List
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt
from itertools import product


PULSAR_COLOR: str = "#805BFF"
NOISE_COLOR: str = "#B6B2B8"
COLORS = [NOISE_COLOR, PULSAR_COLOR]
PALETTE = sns.set_palette(sns.color_palette(COLORS))


def boxplot_comparision(df: pd.DataFrame, features: List[str]):
    rows, cols = 2, 4
    fig, axes = plt.subplots(rows, cols, figsize=(24, 8))
    axes_indexes = list(product(list(range(0, rows)), list(range(0, cols))))

    for idx, feature in enumerate(features):
        index = axes_indexes[idx]
        ax = axes[index[0], index[1]]
        sns.boxplot(
            data=df,
            x="pulsar",
            y=feature,
            ax=ax,
            palette=PALETTE,
            fliersize=1,
            saturation=1,
            linewidth=2.5,
        )

    return fig


def histogram_comparision(df: pd.DataFrame, features: List[str]):
    rows, cols = 2, 4
    fig, axes = plt.subplots(rows, cols, figsize=(24, 8))
    axes_indexes = list(product(list(range(0, rows)), list(range(0, cols))))

    for idx, feature in enumerate(features):
        index = axes_indexes[idx]
        ax = axes[index[0], index[1]]
        ax.hist(
            df[df.pulsar == 0][feature],
            bins=30,
            color=NOISE_COLOR,
            edgecolor="k",
            alpha=0.75,
            label="Ruído",
            density=1,
        )
        ax.hist(
            df[df.pulsar == 1][feature],
            bins=30,
            color=PULSAR_COLOR,
            edgecolor="k",
            alpha=0.85,
            label="Pulsar",
            density=1,
        )
        ax.set_title(f"Distribuição de {feature}")
        ax.legend()

    return fig


def plot_class_balance(df, class_column):
    return (
        df.groupby(class_column)
        .agg({class_column: "count"})
        .rename(columns={class_column: "abs_freq"})
        .assign(rel_freq=lambda df: df.abs_freq / df.abs_freq.sum())
        .style.bar(subset=["rel_freq"])
        .format("{:.2%}", subset=["rel_freq"])
    )
