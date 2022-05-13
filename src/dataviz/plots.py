from itertools import product
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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


def make_bar_chart_comparision(metrics_df: pd.DataFrame, metric: str):
    # Work in progress
    df = metrics_df.copy()
    df["yerr_lower"] = df[metric] - df[f"{metric}_ci_lower"]
    df["yerr_upper"] = df[f"{metric}_ci_upper"] - df[metric]
    models = df["model_name"].unique()
    scenarios = df["scenario_name"].unique()

    colors = ["#6A71EB", "#52EB83", "#EB3E3B", "#EBC946"]

    fig, ax = plt.subplots(dpi=150, figsize=(8, 6))

    bar_width = 0.1
    bar_positions = np.arange(len(scenarios))
    bar_mean_positions = np.zeros(bar_positions.shape)

    for idx, model in enumerate(models):
        bar_mean_positions = bar_mean_positions + bar_positions
        _df = df[df.model_name == model]

        plt.bar(
            bar_positions,
            _df[metric],
            width=bar_width,
            edgecolor="k",
            color=colors[idx],
            label=model,
        )

        plt.errorbar(
            bar_positions,
            _df[metric].tolist(),
            capsize=5,
            yerr=[_df["yerr_lower"], _df["yerr_upper"]],
            fmt="o",
            color="k",
        )
        bar_positions = bar_positions + bar_width

    plt.legend(ncol=2)
    plt.ylabel(metric)
    plt.ylim([0, 1.1])
    plt.title("Comparação entre modelos nos diferentes cenários")

    c_models = len(models)

    if c_models > 1:
        plt.xticks(bar_mean_positions / c_models, scenarios)
    else:
        plt.xticks(bar_positions - bar_width, scenarios)

    plt.savefig(f"outputs/model_comparision-{metric}.png", dpi=300)

    return fig
