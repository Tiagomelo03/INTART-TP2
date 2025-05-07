import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# =============================================
# CONFIGURAÇÕES INICIAIS
# =============================================

plt.style.use('seaborn-v0_8')
sns.set_theme(context="notebook", style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 150)
pd.set_option('display.float_format', '{:.3f}'.format)

PAISES = ['Czech Republic', 'Hungary', 'Poland', 'Slovakia']

SETORES = {
    1: 'Transporte e armazenagem',
    2: 'Comércio atacadista', 
    3: 'Manufatura',
    4: 'Comércio varejista',
    5: 'Energia',
    6: 'Construção civil'
}

# =============================================
# FUNÇÕES AUXILIARES
# =============================================

def carregar_dados(anos):
    dfs = []
    for ano in anos:
        np.random.seed(42)
        dados = pd.DataFrame(np.random.randn(450, 82).cumsum(axis=0),
                             columns=[f'X{i}' for i in range(1, 83)])
        dados['Country'] = np.random.choice(PAISES, 450)
        dados['S'] = np.random.choice(list(SETORES.keys()), 450)
        dados['ano'] = ano
        for col in dados.columns[:20]:
            dados.loc[dados.sample(frac=0.1).index, col] = np.nan
        dfs.append(dados)
        print(f"Dados {ano} simulados - {len(dados)} registros")
    return pd.concat(dfs, ignore_index=True)

def estatisticas_descritivas(df, cols, grupo=None):
    if grupo:
        desc = df.groupby(grupo)[cols].describe().T
    else:
        desc = df[cols].describe().T
    if grupo:
        modas = df.groupby(grupo)[cols].agg(lambda x: x.mode()[0])
    else:
        modas = df[cols].mode().iloc[0]
    desc['moda'] = modas
    return desc.round(3)

def intervalo_confianca(series, conf=0.95):
    n = series.count()
    m = series.mean()
    se = stats.sem(series)
    h = se * stats.t.ppf((1 + conf) / 2., n-1)
    return (m - h, m + h)

# MELHORIA 1: IC para a mediana usando bootstrapping
def intervalo_confianca_mediana(series, conf=0.95, n_boot=1000):
    boot_meds = np.random.choice(series, (n_boot, len(series)), replace=True).mean(axis=1)
    low = np.percentile(boot_meds, (1 - conf) / 2 * 100)
    high = np.percentile(boot_meds, (1 + conf) / 2 * 100)
    return (low, high)

# MELHORIA 2: Tratamento robusto do teste de normalidade
def analisar_normalidade(df, cols):
    resultados = {}
    for col in cols:
        try:
            if df[col].nunique() > 3:  # Evita colunas constantes ou quase constantes
                stat, p = stats.shapiro(df[col])
                resultados[col] = {'Estatística': stat, 'p-valor': p}
            else:
                resultados[col] = {'Estatística': np.nan, 'p-valor': np.nan}
        except Exception as e:
            resultados[col] = {'Estatística': np.nan, 'p-valor': np.nan}
    return pd.DataFrame(resultados).T

# =============================================
# ANÁLISE PRINCIPAL
# =============================================

def analise_completa():
    print("="*50)
    print(" ANÁLISE ESTATÍSTICA - DADOS FINANCEIROS ")
    print("="*50 + "\n")

    df = carregar_dados(['2017', '2018', '2019', '2020'])
    variaveis = [f'X{i}' for i in range(1, 83)]

    print("\nPré-processamento...")
    imputer = SimpleImputer(strategy='median')
    df[variaveis] = imputer.fit_transform(df[variaveis])

    print("\nEstatísticas Globais (Top 10 Variáveis):")
    top_vars = df[variaveis].var().sort_values(ascending=False).index[:10]
    print(estatisticas_descritivas(df, top_vars))

    print("\nEstatísticas por País (Top 5 Variáveis):")
    for pais in PAISES:
        print(f"\n--- {pais} ---")
        subset = df[df['Country'] == pais]
        print(estatisticas_descritivas(subset, top_vars[:5]))

    print("\nEstatísticas por Setor Econômico:")
    for cod, nome in SETORES.items():
        print(f"\n--- {nome} (Código {cod}) ---")
        subset = df[df['S'] == cod]
        print(estatisticas_descritivas(subset, top_vars[:5]))

    print("\nTestando Normalidade (Shapiro-Wilk) nas Top Variáveis:")
    print(analisar_normalidade(df, top_vars))

    print("\nIntervalos de Confiança (95%) para a MÉDIA:")
    for col in top_vars:
        ic = intervalo_confianca(df[col])
        print(f"{col}: IC95% (Média) = ({ic[0]:.3f}, {ic[1]:.3f})")

    print("\nIntervalos de Confiança (95%) para a MEDIANA:")
    for col in top_vars:
        ic_med = intervalo_confianca_mediana(df[col])
        print(f"{col}: IC95% (Mediana) = ({ic_med[0]:.3f}, {ic_med[1]:.3f})")

    print("\nGerando visualizações...")

    plt.figure(figsize=(12, 10))
    sns.heatmap(df[top_vars].corr(), annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Matriz de Correlação (Top 10 Variáveis)')
    plt.show()

    pca = PCA(n_components=2)
    componentes = pca.fit_transform(df[top_vars])
    df['PC1'] = componentes[:, 0]
    df['PC2'] = componentes[:, 1]

    # MELHORIA 3: Variância explicada pelo PCA
    print("\nVariância Explicada pelas Componentes Principais:")
    for i, var_ratio in enumerate(pca.explained_variance_ratio_):
        print(f"Componente {i+1}: {var_ratio*100:.2f}%")

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='PC1', y='PC2', hue='Country', palette='viridis')
    plt.title('Análise de Componentes Principais por País')
    plt.show()

    print("\nScatterplot Matrix das Top Variáveis:")
    sns.pairplot(df[top_vars[:5]])
    plt.show()

    print("\nAtributos mais importantes segundo PCA (PC1):")
    importancia = pd.Series(np.abs(pca.components_[0]), index=top_vars)
    print(importancia.sort_values(ascending=False))

if __name__ == "__main__":
    analise_completa()
