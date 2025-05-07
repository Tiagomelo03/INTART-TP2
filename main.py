import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pandas.plotting import scatter_matrix

# Carregar o CSV corretamente (com separador ; e decimal ,)
df = pd.read_csv('Insol.csv', sep=';', decimal=',', on_bad_lines='skip')

# Visão geral
print("Dimensões do dataset:", df.shape)
print("Colunas:", df.columns.tolist())
print("\nPrimeiras linhas:\n", df.head())

# Tipos de dados
print("\nTipos de dados:\n", df.dtypes)

# Estatísticas básicas
print("\nResumo estatístico geral:\n", df.describe(include='all'))

# Moda
print("\nModa:\n", df.mode().iloc[0])

# Mediana
print("\nMediana:\n", df.median(numeric_only=True))

# Desvio padrão
print("\nDesvio padrão:\n", df.std(numeric_only=True))

# Quartis
print("\n1º Quartil (Q1):\n", df.quantile(0.25, numeric_only=True))
print("\n3º Quartil (Q3):\n", df.quantile(0.75, numeric_only=True))


# Scatter Matrix (primeiras 5 colunas numéricas para visualização)
cols_numericas = df.select_dtypes(include=[np.number]).columns[:5]
scatter_matrix(df[cols_numericas], figsize=(10, 8))
plt.suptitle("Scatter Matrix - primeiras 5 variáveis numéricas")
plt.savefig("scatter_matrix.png")
plt.close()

# Histograma + Curva de densidade para cada coluna numérica (primeiras 5)
for col in cols_numericas:
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribuição de {col}')
    plt.savefig(f"distribuicao_{col}.png")
    plt.close()

# Teste de normalidade (Shapiro-Wilk)
for col in cols_numericas:
    amostra = df[col].dropna()
    stat, p = stats.shapiro(amostra)
    print(f"Shapiro-Wilk para {col} → p-value: {p:.4f} {'(Segue distribuição normal)' if p > 0.05 else '(Não segue distribuição normal)'}")

# Intervalo de confiança da média da primeira coluna numérica
col = cols_numericas[0]
sample = df[col].dropna()
mean = np.mean(sample)
conf_int = stats.t.interval(confidence=0.95, df=len(sample)-1, loc=mean, scale=stats.sem(sample))
print(f"\nIntervalo de confiança 95% para a média de '{col}': {conf_int}")
