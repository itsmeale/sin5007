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

# # Características do dataset e pré-processamento

# ---

# # Características e descrição do problema
#
# ### Descrição do problema
#
# Pulsares são estrelas de nêutrons que giram rapidamente e emitem feixes de ondas de rádio. \
# Detectar essas estrelas é uma tarefa relevante para várias aplicações da astronomia, como:
# - Cronometragem de alta precisão
# - Detecção de planetas extrasolares
# - Mecânica celeste e astrometria
# - Estudos de matéria superdensa
# - Física gravitacional no regime de campo forte 
# - Física de plasma sob condições extremas
#
# Outras aplicações podem ainda ser encontradas em algumas referências como:
# ```citation
# LORIMER, D. R., & KRAMER, M. (2005). Handbook of pulsar astronomy. Cambridge, UK, Cambridge University Press.
# ```
#
# ### Variáveis
# O dataset é composto por 8 variáveis independentes e 1 variável alvo que é a classificação do pulsar.
#
# #### Variáveis independentes
# As variáveis independentes são todas quantitativas contínuas, mas variam as suas classificações entre variáveis intervalares e racionais:
# - média do perfil integrado: **racional** (zero absoluto = 0, significa ausência de sinal de rádio, unidade de medida é W/m²)
# - desvio padrão do perfil integrado: **intervalar**
# - curtose do perfil integrado: **intervalar**
# - assimetria do perfil integrado: **intervalar**
# - media da curva dm-snr: **racional** (zero absoluto = 0, significa ausência de sinal de rádio)
# - desvio padrão da curva dm-snr: **intervalar**
# - curtose da curva dm-snr: **intervalar**
# - assimetria da curva dm-snr: **intervalar**
#
# #### Variável dependente
# As classes são dividídas em 2:
# - 1: classificado como **pulsar**
# - 0: classificado como **ruído**
#
# ### Instâncias e balanceamento das classes
#
# - Número total de instâncias: 17.898
# - Classe 0 (ruído): 16.259 (90,84%)
# - Classe 1 (pulsar): 1.639 (9,16%)
#
# ### Instâncias sem missing values
# - 17.898, não há missing values no dataset.
#
#
# ### Glossário
# - **Phase**: Período ou fase de rotação do pulsar
# - **Perfil Integrado**: Vetor de intensidade do sinal de rádio nos diferentes períodos de rotação do pulsar (W/m²)
# - **DM-SNR**: Vetor da razão de sinal-ruído para diferentes valores de medidas de dispersão
# - **SNR**: Razão de intensidade de sinal para ruído (Signal to Noise Ratio)
# - **DM**: Medida de dispersão (Dispersion Measure)

# ---

# # Bibliotecas

# +
from itertools import product

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from src.dataviz.plots import histogram_comparision, boxplot_comparision, plot_class_balance
# -

# # Dataset

DATASET_PATH: str = "../data/raw/HTRU_2.csv"
TRAIN_SET_PATH: str = "../data/raw/dev_set.csv"

df = pd.read_csv(DATASET_PATH)

# ---

# ## Shape

print("Instâncias totais:", df.shape[0])

print("Distribuição das classes")
plot_class_balance(df, "pulsar")

# ## Missing

df.isna().sum()

# Nenhuma feature com valores nulos.

# ---

# # Análise exploratória de dados

# Distribuição das características entre as duas classes

features = df.columns.tolist()[:-1]

df.describe()

df.groupby("pulsar").median()

histogram_comparision(df, features)
plt.show()

# + tags=[]
boxplot_comparision(df, features)
plt.show()
# -

# ## Teste de correlação

sns.pairplot(df, hue="pulsar", hue_order=[0, 1], corner=True)

# ## Teste de normalidade das features

from scipy.stats import normaltest

# A hipótese nula é a de que a variável tem uma distribuição normal, \
# rejeitar a hipótese nula significa assumir que as variáveis não possuem uma \
# distribuição normal.

# +
test_results = list()

for feature in features:
    pvalue = normaltest(df[feature]).pvalue
    test_results.append((feature, pvalue))

normality_test_df = (
    pd.DataFrame(test_results, columns=["feature", "pvalue"])
    .assign(is_normal=lambda df: df.pvalue > 5e-2)
)
# -

normality_test_df

# Tentativa de correção de assimetrias e achatamento

# +
test_results = list()
only_noise_df = df[df.pulsar==0]

operations = [
    (lambda x: 1/x, "1/x"),  # distribuicoes achatadas
    (lambda x: np.log(x), "log"),  # assimetrias positivas
    (lambda x: np.sqrt(x), "sqrt"),  # assimetrias positivas
    (lambda x: x**2, "x**2"),  #  assimetrias negativas
    (lambda x: x**3, "x**3"),  # assimetrias negativas
]

for op, name_op in operations:
    for feature in features:
        values = df[feature].apply(op)
        pvalue = normaltest(values.fillna(values.mean())).pvalue
        test_results.append((feature, name_op, pvalue))

normality_test_df = (
    pd.DataFrame(test_results, columns=["feature", "operation", "pvalue"])
    .assign(is_normal=lambda df: df["pvalue"] > 5e-2)
)
# -

normality_test_df[normality_test_df.is_normal]

# ## Testes de hipóteses baseados na literatura

from scipy.stats import mannwhitneyu

pulsar = df[df.pulsar==1]
not_pulsar = df[df.pulsar==0]

# ### A razão SNR é maior nos pulsares

mannwhitneyu(pulsar["dmsnr_media"], not_pulsar["dmsnr_media"]).pvalue

mannwhitneyu(pulsar["dmsnr_media"], not_pulsar["dmsnr_media"]).pvalue < 5e-2

df.groupby("pulsar")["dmsnr_media"].mean()

# ## Notas da EDA
#
# - Nenhuma das features é normal e não há uma transformação trivial para ajustar as assimetrias das distribuições
# - Pulsares tem SNR maior que o ruído.
# - O Ruído tem uma assimetria positiva maior do que os pulsares, provavelmente pois o maior SNR se concentram em baixos valores de DM, o que já era esperado pela literatura.
# - A intensidade média do sinal dos pulsares tem uma distribuição mais uniforme, enquanto a do ruído é mais "previsível", de fato, cada pulsar tem momentos diferentes de pico de intensidade baseado em sua rotação.
# - A intensidade média do sinal dos pulsares tende a ser menor que a intensidade média do sinal de ruído (talvez pela distância dos objetos?).

# ---

# # Etapas de pré-processamento
# - Eliminação de ruídos
# - Normalização
# - Balanceamento de classes

# ## Remoção de outliers

df = pd.read_csv(TRAIN_SET_PATH)

print("Distribuição das classes")
plot_class_balance(df, "pulsar")

df_without_outliers = df.copy()


def remove_outliers(df, column):
    q1 = df[column].quantile(.25)
    q3 = df[column].quantile(.75)
    iqr = q3 - q1
    lower = q1 - (1.5 * iqr)
    upper = q3 + (1.5 * iqr)
    return df[(df[column] >= lower) & (df[column] <= upper)]


for feature in features:
    df_without_outliers = remove_outliers(df_without_outliers, feature)

print("Distribuição das classes")
plot_class_balance(df_without_outliers, "pulsar")

# +
from sklearn.ensemble import IsolationForest

df2 = df.copy()
clf = IsolationForest()
clf.fit(df2)
df2["outlier"] = clf.predict(df2)
df2 = df2[df2.outlier!=-1]

print("Distribuição das classes")
plot_class_balance(df2, "pulsar")

# + tags=[]
from sklearn.neighbors import LocalOutlierFactor

df3 = df.copy()
clf = LocalOutlierFactor()
# clf.fit(df3)
df3["outlier"] = clf.fit_predict(df3)
df3 = df3[df3.outlier!=-1]

print("Distribuição das classes")
plot_class_balance(df3, "pulsar")
# -

df_without_outliers = df3.copy().drop(columns=["outlier"])

# ---

# ## Normalização

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df_without_outliers.loc[:, features] = scaler.fit_transform(df_without_outliers[features])

df_without_outliers.head(3)

# +
import joblib

joblib.dump(scaler, "../models/preprocessing/minmax_scaler.joblib")
# -

# ---

# ## Balanceamento das classes

from imblearn.over_sampling import SMOTE

# +
smote = SMOTE()

X, y = df_without_outliers[features], df_without_outliers["pulsar"]
df_balanced, y_new = smote.fit_resample(df_without_outliers[features], df_without_outliers["pulsar"])
# -

df_balanced["pulsar"] = y_new

print("Distribuição das classes")
plot_class_balance(df_balanced, "pulsar")

print("Distribuição das classes")
plot_class_balance(df_without_outliers, "pulsar")

# ---

# ## Train sets pré-processados
# - dataset com tratamento de outliers, com padronização e sem balanceamento de classes
# - dataset com tratamento de outliers, com padronização e com balanceamento de classes

df_without_outliers.to_csv("../data/preprocessed/train_unbalanced.csv", index=False)
df_balanced.to_csv("../data/preprocessed/train_balanced.csv", index=False)

df_without_outliers.head(10)

df_balanced.shape


