# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ![image.png](attachment:9ec502cf-dfeb-40b6-8bf5-3c6ddf605c12.png)

# # Análise de componentes principais

# ---

# # Bibliotecas

# +
import numpy as np
import pandas as pd
import seaborn as sns
from src.dataviz.plots import plot_class_balance
import matplotlib.pyplot as plt

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
# -

# # Dataset

DATASET_PREPROCESSED_PATH: str = "../data/preprocessed/HTRU_2_outliers_removed.csv"
DATASET_RAW_PATH: str = "../data/raw/HTRU_2.csv"

# 2 datasets foram preparados para testar a redução de dimensionalidade com PCA \
# É sábido que o PCA, utilizando matriz de covariância, é sensível a diferentes escalas e unidades de medida, \
# No entanto, ao utilizar PCA com a matriz de correlação essas variações são menos significativas, pois a correlação \
# normaliza as variáveis por estar contida num intervalo específico [-1, 1] para todas as features.
#
# Nesta implementação, utilizaremos a biblioteca scikit-learn do Python que implementa o PCA utilizando SVD (Single Value Decomposition). \
# O SVD é uma técnica de decomposição de matrizes, se formos comparar esta técnica com a determinação dos autovalores e autovetores através das matrizes de covariância e correlação, \
# os resultados obtidos através do SVD serão similares a aplicação utilizando a matriz de covariância, dado que não há garantias de intervalos bem comportados para todas as variáveis \
# do dataset.
#
# Por isso, esperamos que a aplicação de PCA usando SVD nos dados normalizados (utilizando min-max scaling), trará resultados mais interessantes, pois não haverá domínio das variáveis
# de unidades de medida e escalas maiores.

df = pd.read_csv(DATASET_RAW_PATH)
df_wo = pd.read_csv(DATASET_PREPROCESSED_PATH)

# Sabemos que a proporção da variância explicada pelos componentes principais é dada pela equação:
# $$
# p_j = \frac{\lambda_j}{\sum_i^n{\lambda_i}}
# $$
#
# Onde $\lambda_i$ é a variância explicada pelo componente $i$.
#
# Pela formulação do PCA utilizando a matriz de covariância, nos é dado que os primeiros PCs necessariamente serão dominados pelas \
# variáveis com a maior variância, e estando as variáveis em escalas diferentes, os primeiros PCs podem fácilmente capturar quase toda \
# proporção da variância explicada.
#
# Para reduzir este efeito da variância, utilizamos um dataset com a remoção de outliers através da técnica LocalOutlierFactor que trabalha de \
# forma multivariada a remoção de outliers.

print(df.shape)
print(df_wo.shape)

# O dataset $df_{wo}$ é o conjunto de dados com os outliers removidos, como podemos perceber, poucas observações foram removidas.

df.var()

df_wo.var()

# Além disso, percebemos também que ouve pouco efeito sobre a variância das variáveis. Portanto, não há muitas expectativas de que a remoção de outlier
# por si só, seja suficiente para mitigar o problema da escala das variáveis.

print("Dataset original")
df.describe().loc[["std"]].T.sort_values(by=["std"], ascending=False)

print("Dataset original")
df_wo.describe().loc[["std"]].T.sort_values(by=["std"], ascending=False)

print("Dataset original")
df.describe().loc[["min", "max", "std"]]

print("Dataset sem outliers")
df_wo.describe().loc[["min", "max", "std"]]

pd.DataFrame(MinMaxScaler().fit_transform(df), columns=df.columns).describe().loc[["std"]].drop(columns="pulsar").T.sort_values(by=["std"], ascending=False)

# ## Shape

print("Distribuição das classes")
plot_class_balance(df, "pulsar")

print("Distribuição das classes")
plot_class_balance(df_wo, "pulsar")

# Em trabalhos anteriores, já haviamos identificado que existem variáveis altamente correlacionadas no nosso conjunto de dados, \
# como as relações **perfil_integrado_assimetria-perfil_integrado_curtose**, **dmsnr_assimetria-dmsnr_curtose** e **dm_snr_curtose-dm_snr_desvio**.
#
# Sendo assim, há indícios de pelo menos 3 variáveis (ou componentes) que são redundantes.

fig ,ax = plt.subplots(figsize=(12, 6))
sns.heatmap(df.corr("pearson"), annot=True, ax=ax, linewidths=.5, mask=np.triu(df.corr()))
plt.show()

# ---

# ## Testes com PCA

X1 = df.drop(columns=["pulsar"])
X2 = df_wo.drop(columns=["pulsar"]).values

# ### PCA puro

pca_1 = PCA()
pca_1.fit(X1)

# ### PCA puro WO

pca_2 = PCA()
pca_2.fit(X2)

# ### PCA com normalização

pipe = Pipeline(steps=[
    ("minmaxscaler", MinMaxScaler()),
    ("pca", PCA()),
])

pipe.fit(X1)

pipe2 = Pipeline(steps=[
    ("minmaxscaler", MinMaxScaler()),
    ("pca", PCA()),
])
pipe2.fit(X2)

from imblearn.pipeline import Pipeline as Pipe2
from imblearn.over_sampling import SMOTE

pipe3 = Pipe2(steps=[
    ("smt", SMOTE()),
    ("scaling", MinMaxScaler()),
    ("pca", PCA())
])
pipe3.fit(X1, df["pulsar"])

pca_3 = pipe.steps[1][1]
pca_4 = pipe2.steps[1][1]
pca_5 = pipe3.steps[2][1]

print("Componentes principais")
pd.DataFrame(pca_3.components_, columns=df.drop(columns=["pulsar"]).columns).T

print("Variância acumulada")
pd.DataFrame(np.cumsum(pca_3.explained_variance_ratio_)).T

fig, ax = plt.subplots(figsize=(8, 6))
plt.plot(list(range(len(pca_1.explained_variance_ratio_))), np.cumsum(pca_1.explained_variance_ratio_), ".r-", markersize=10, label="PCA com outlier")
plt.plot(list(range(len(pca_2.explained_variance_ratio_))), np.cumsum(pca_2.explained_variance_ratio_), "dg--", markersize=10, label="PCA sem outliers")
plt.plot(list(range(len(pca_3.explained_variance_ratio_))), np.cumsum(pca_3.explained_variance_ratio_), "b.-", markersize=10, label="PCA com outliers variáveis normalizadas")
plt.plot(list(range(len(pca_4.explained_variance_ratio_))), np.cumsum(pca_4.explained_variance_ratio_), "k-", markersize=10, label="PCA sem outliers variáveis normalizadas")
plt.xlabel("Componentes")
plt.ylabel("Variância total explicada acumulada")
plt.ylim(0, 1.1)
plt.title("Comparação da proporção acumulada da variância explicada\npara diferentes técnicas de pré-processamento com PCA")
plt.legend()
plt.savefig("myfig.png", dpi=300)


fig, ax = plt.subplots(figsize=(8, 6))
plt.plot(list(range(len(pca_3.explained_variance_ratio_))), pca_3.explained_variance_ratio_, "b.-", markersize=10, label="PCA com outliers variáveis normalizadas")
plt.xlabel("Componente")
plt.ylabel("Variância total explicada")
plt.ylim(0, 1.1)
plt.title("Seleção de componentes através do 'cotovelo' da curva (scree graph)")
plt.legend()
plt.savefig("myfig2.png", dpi=300)


# Definiremos o número de componentes a serem utilizados como sendo os k primeiros componentes cuja variância explicada acumulada seja >= 95%. \
# O limiar de 95% é definido com base no argumento apresentado por JOLLIFFE (Principal Component Analysis, 2002) de que, quando os primeiros componentes são muito dominantes, um limiar \
# maior que 90% pode ser necessário para capturar estruturas menos óbvias.

k = np.where(np.cumsum(pca_3.explained_variance_ratio_) > 0.95)[0][0]+1

k

pipe = Pipeline(steps=[
    ("minmaxscaler", MinMaxScaler()),
    ("pca", PCA(n_components=5)),
])

df_2plot = pd.DataFrame(pipe.fit_transform(df), columns=["x1", "x2", "x3", "x4", "x5"])
df_2plot["pulsar"] = df["pulsar"].map({1: "pulsar", 0: "noise"})

import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'iframe'

fig = px.scatter_3d(df_2plot, x='x1', y='x2', z='x3', color="pulsar", symbol="pulsar")
fig.show()

from sklearn.ensemble import AdaBoostClassifier

pipe = Pipe2(steps=[
    ("minmaxscaler", MinMaxScaler()),
    ("pca", PCA(n_components=5)),
    ("clf", AdaBoostClassifier())
])

pipe2 = Pipe2(steps=[
    ("minmaxscaler", MinMaxScaler()),
    ("clf", AdaBoostClassifier())
])

from sklearn.model_selection import cross_validate

X = df.iloc[:, :-1]
y = df["pulsar"]

d = cross_validate(pipe, X, y, scoring=["accuracy", "precision", "recall", "f1"], cv=10)
e = cross_validate(pipe2, X, y, scoring=["accuracy", "precision", "recall", "f1"], cv=10)

for key, value in d.items():
    print(key, value.mean(), value.std())

for key, value in e.items():
    print(key, value.mean(), value.std())
