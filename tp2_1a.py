import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# ====================================================
# Configurações e Ajustes de Display
# ====================================================
# Caminho da pasta com os CSVs
pasta = os.path.dirname(os.path.abspath(__file__))
 # Ajuste se necessário
# Configurações para exibição tabular truncada com ellipsis
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)

# ====================================================
# Função: Carregar e Concatenar Dados
# ====================================================
def carregar_dados(pasta):
    """
    Lê todos os arquivos .csv da pasta, trata 'm' como NaN,
    e retorna um DataFrame único.
    """
    arquivos = [f for f in os.listdir(pasta) if f.lower().endswith('.csv')]
    lista_df = []

    for arquivo in sorted(arquivos):
        caminho = os.path.join(pasta, arquivo)
        try:
            df = pd.read_csv(caminho, na_values='m')
            df['Ano'] = int(arquivo[:4])
            lista_df.append(df)
            print(f"✅ {arquivo}: {df.shape[0]} linhas, {df.shape[1]} colunas")
        except Exception as e:
            print(f"❌ Erro em {arquivo}: {e}")

    if not lista_df:
        raise RuntimeError("Nenhum CSV carregado com sucesso.")

    df_total = pd.concat(lista_df, ignore_index=True)
    print(f"\n📦 Total combinado: {df_total.shape[0]} linhas, {df_total.shape[1]} colunas")
    return df_total

# ====================================================
# Função: Pré-processamento
# ====================================================
def preprocessar(df):
    """
    Imputa NAs pela mediana e garante tipos corretos.
    """
    imputer = SimpleImputer(strategy='median')
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = imputer.fit_transform(df[num_cols])

    for col in ['Country', 'S', 'Ano']:
        if col in df.columns:
            df[col] = df[col].astype('category')
    return df

# ====================================================
# Função: Estatísticas Descritivas
# ====================================================
def estatisticas(df, grupo=None):
    """
    Retorna DataFrame de estatísticas descritivas.
    Se grupo for fornecido, agrupa antes de resumir.
    Para agrupamentos, empilha variáveis para reduzir número de colunas,
    deixando métricas (count, mean, std, min, 25%, 50%, 75%, max) como colunas.
    """
    if grupo:
        # Estatísticas por grupo: resultado com colunas MultiIndex (var, stat)
        # Adicionando observed=True para resolver o aviso
        descr = df.groupby(grupo, observed=True).describe()
        # Empilha nível de variável, mantendo estatísticas como colunas
        # Adicionando future_stack=True para resolver o aviso
        descr = descr.stack(level=0, future_stack=True)
        # Renomeia índices para ficar claro
        descr.index.names = list(descr.index.names)  # ex: ['S','variable'] or ['S','Country','variable']
        return descr
    else:
        # Estatísticas gerais: métricas como colunas, variáveis como índice
        descr = df.describe().T
        return descr

# ====================================================
# Função: Teste Normalidade e Intervalo de Confiança
# ====================================================
def teste_normalidade_e_ic(df, col, conf=0.95):
    dados = df[col].dropna()
    stat, p = stats.shapiro(dados)
    interpret = (f"A variável {col} parece seguir uma distribuição normal" if p > 0.05
                 else f"A variável {col} NÃO segue distribuição normal")
    interpret += f" (p = {p:.3e})."

    n = len(dados)
    m = np.mean(dados)
    se = stats.sem(dados)
    margem = se * stats.t.ppf((1 + conf) / 2., n - 1)
    ic_lower, ic_upper = m - margem, m + margem
    ic_text = (f"Intervalo de confiança de {conf*100:.0f}% para a média de {col}: "
               f"[{ic_lower:.4f}, {ic_upper:.4f}].")
    return interpret, ic_text

# ====================================================
# Funções de Gráficos
# ====================================================
def plot_heatmap(df):
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.select_dtypes(include=[np.number]).corr(),
                cmap='coolwarm', center=0)
    plt.title('Heatmap de Correlação')
    plt.show()

def plot_scatter_matrix(df, cols, hue='S'):
    sample = df.sample(n=min(200, len(df)), random_state=42)
    sns.pairplot(sample[cols + [hue]], hue=hue)
    plt.show()

# ====================================================
# Função: Importância de Atributos
# ====================================================
def atributos_importantes(df, target='S', top_n=10):
    X = df.drop(columns=[target, 'Country', 'Ano', 'Num'], errors='ignore')
    X = X.select_dtypes(include=[np.number])
    y = df[target].cat.codes
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X_train, y_train)
    importancias = pd.Series(clf.feature_importances_, index=X.columns)
    return importancias.sort_values(ascending=False).head(top_n)

# ====================================================
# Rotina Principal
# ====================================================
if __name__ == '__main__':
    df = carregar_dados(pasta)
    df = preprocessar(df)

    if 'X1' in df.columns:
        df = df.drop(columns=['X1'])

    print("\n" + "="*60)
    print("📊 ESTATÍSTICAS GERAIS")
    print("="*60)
    print(estatisticas(df))

    print("\n" + "="*60)
    print("📊 ESTATÍSTICAS POR SETOR (S)")
    print("="*60)
    print(estatisticas(df, grupo='S'))

    print("\n" + "="*60)
    print("🔥 HEATMAP DE CORRELAÇÃO")
    print("="*60)
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.select_dtypes(include=[np.number]).corr(),
                cmap='coolwarm', center=0)
    plt.title('Heatmap de Correlação')
    plt.tight_layout()
    plt.savefig("heatmap_correlacao.png")
    plt.close()
    print("📸 Guardado como: heatmap_correlacao.png")

    print("\n" + "="*60)
    print("📷 SCATTERPLOT MATRIX (X2 a X5)")
    print("="*60)
    sample = df.sample(n=min(200, len(df)), random_state=42)
    sns.pairplot(sample[['X2', 'X3', 'X4', 'X5', 'S']], hue='S')
    plt.savefig("scatterplot_matrix.png")
    plt.close()
    print("📸 Guardado como: scatterplot_matrix.png")

    print("\n" + "="*60)
    print("⭐ TOP 10 ATRIBUTOS MAIS IMPORTANTES")
    print("="*60)
    top_atributos = atributos_importantes(df)
    print(top_atributos)

    # Gráfico dos atributos mais importantes
    top_atributos.plot(kind='barh', title='Top 10 Atributos Mais Importantes')
    plt.xlabel("Importância")
    plt.tight_layout()
    plt.savefig("grafico_top_atributos.png")
    plt.close()
    print("📸 Guardado como: grafico_top_atributos.png")

    print("\n📌 CONCLUSÃO MANUAL (exemplo):")
    print("- A maioria das variáveis apresenta distribuição assimétrica.")
    print("- Existem fortes correlações entre certas variáveis (ver heatmap).")
    print("- X50, X47 e X24 são os atributos mais relevantes para o target 'S'.")

    print("\n✅ Análise estatística e visual finalizada com sucesso!")
