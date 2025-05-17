import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import RFE
import warnings
warnings.filterwarnings('ignore')

# Configurações de visualização
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# 1. Função para carregar os dados
def carregar_dados(caminho_arquivo):
    try:
        dados = pd.read_csv(caminho_arquivo)
        print(f"Dados carregados com sucesso. Formato: {dados.shape}")
        return dados
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        return None

# 2. Função de pré-processamento
def preprocessar_dados(df):
    df_proc = df.copy()
    
    # Converter valores 'm' para NaN e depois para numérico
    for col in df_proc.columns:
        if df_proc[col].dtype == 'object':
            df_proc[col] = df_proc[col].replace('m', np.nan)
            df_proc[col] = pd.to_numeric(df_proc[col], errors='ignore')
    
    # Remover outliers da variável alvo X44
    if 'X44' in df_proc.columns:
        Q1 = df_proc['X44'].quantile(0.25)
        Q3 = df_proc['X44'].quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        df_proc = df_proc[(df_proc['X44'] >= limite_inferior) & (df_proc['X44'] <= limite_superior)]
        print(f"Outliers removidos de X44. Novo shape: {df_proc.shape}")
    
    return df_proc

# 3. Análise exploratória
def analise_exploratoria(df, variavel_alvo):
    print(f"\n=== Análise Exploratória: {variavel_alvo} ===")
    print(df[variavel_alvo].describe())

    # Gráficos de distribuição
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(df[variavel_alvo].dropna(), kde=True, bins=50)
    plt.title(f'Distribuição de {variavel_alvo}')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(y=df[variavel_alvo].dropna(), showfliers=False)
    plt.title(f'Boxplot de {variavel_alvo}')
    plt.tight_layout()
    plt.show()

    # Análise de correlação
    colunas_numericas = df.select_dtypes(include=np.number).columns
    correlacoes = df[colunas_numericas].corr()[variavel_alvo].sort_values(ascending=False)
    
    print("\nTop 10 correlações com a variável alvo:")
    print(correlacoes.head(10))
    
    # Gráfico das correlações
    plt.figure(figsize=(10, 6))
    correlacoes[1:11].plot(kind='barh')  # Exclui a própria variável alvo
    plt.title('Top 10 Correlações com a Variável Alvo')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return correlacoes

# 4. Preparação para modelagem
def preparar_dados_modelagem(df, variavel_alvo, colunas_selecionadas=None, normalizacao=True):
    df_model = df.dropna(subset=[variavel_alvo])
    
    if colunas_selecionadas is None:
        colunas_numericas = df_model.select_dtypes(include=np.number).columns.tolist()
        colunas_numericas.remove(variavel_alvo)
        colunas_selecionadas = colunas_numericas
    
    # Verificar e remover valores ausentes
    na_count = df_model[colunas_selecionadas].isna().sum()
    print("\nValores ausentes nas features:")
    print(na_count[na_count > 0])
    
    df_model = df_model.dropna(subset=colunas_selecionadas)
    print(f"\nDados para modelagem: {df_model.shape}")
    
    X = df_model[colunas_selecionadas]
    y = df_model[variavel_alvo]
    
    if normalizacao:
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    return X, y

# 5. Seleção de features
def selecionar_features(X, y, n_features=15, modelo=LinearRegression()):
    rfe = RFE(estimator=modelo, n_features_to_select=n_features)
    rfe.fit(X, y)
    
    features_selecionadas = X.columns[rfe.support_]
    print(f"\nFeatures selecionadas ({len(features_selecionadas)}):")
    print(features_selecionadas.tolist())
    
    return features_selecionadas

# 6. Função de regressão linear (AJUSTADA)
def treinar_regressao_linear(X, y):
    # Dividir os dados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Treinar modelo
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    
    # Previsões
    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)
    
    # Métricas
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    
    # Validação cruzada
    cv_scores = cross_val_score(modelo, X, y, cv=5, scoring='r2')
    
    # Resultados
    print("\n=== Resultados da Regressão Linear ===")
    print(f"{'Métrica':<20} {'Treino':<10} {'Teste':<10}")
    print(f"{'RMSE':<20} {rmse_train:<10.2f} {rmse_test:<10.2f}")
    print(f"{'R²':<20} {r2_train:<10.4f} {r2_test:<10.4f}")
    print(f"\nR² Validação Cruzada (5 folds): {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    # Coeficientes
    coeficientes = pd.DataFrame({
        'Feature': X.columns,
        'Coeficiente': modelo.coef_
    }).sort_values(by='Coeficiente', ascending=False)
    
    print("\nTop 10 Coeficientes Positivos:")
    print(coeficientes.head(10))
    
    print("\nTop 10 Coeficientes Negativos:")
    print(coeficientes.tail(10))
    
    # Visualização dos coeficientes
    plt.figure(figsize=(12, 6))
    top_coef = pd.concat([coeficientes.head(10), coeficientes.tail(10)])
    sns.barplot(x='Coeficiente', y='Feature', data=top_coef)
    plt.title('Top 20 Coeficientes (Positivos e Negativos)')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Gráfico de previsões vs reais
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred_test, alpha=0.5)
    lim_min = min(y_test.min(), y_pred_test.min())
    lim_max = max(y_test.max(), y_pred_test.max())
    plt.plot([lim_min, lim_max], [lim_min, lim_max], 'r--')
    plt.xlabel('Valores Reais')
    plt.ylabel('Previsões')
    plt.title('Previsões vs Valores Reais')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Análise de resíduos
    residuos = y_test - y_pred_test
    plt.figure(figsize=(10, 6))
    sns.histplot(residuos, kde=True, bins=30)
    plt.title('Distribuição dos Resíduos')
    plt.xlabel('Resíduos (y_real - y_pred)')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return modelo, coeficientes

# 7. Criar variável binária
def criar_variavel_binaria(df, coluna, threshold=0):
    nova_coluna = f'{coluna}_bin'
    df[nova_coluna] = (df[coluna] > threshold).astype(int)
    print(f"\nDistribuição da variável binária ({nova_coluna}):")
    print(df[nova_coluna].value_counts(normalize=True) * 100)
    return nova_coluna

# 8. Regressão logística
def treinar_regressao_logistica(X, y, setor_filtro=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    modelo = LogisticRegression(max_iter=1000, random_state=42)
    modelo.fit(X_train, y_train)
    
    y_pred = modelo.predict(X_test)
    
    print(f"\n=== Resultados da Regressão Logística {f'(Setor {setor_filtro})' if setor_filtro else ''} ===")
    print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}")
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))
    
    # Matriz de confusão
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), 
                annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negativo', 'Positivo'],
                yticklabels=['Negativo', 'Positivo'])
    plt.title('Matriz de Confusão')
    plt.xlabel('Previsto')
    plt.ylabel('Real')
    plt.show()
    
    return modelo

# 9. Função principal
def executar_analise():
    # Carregar dados
    dados = carregar_dados('dados_2017_a_2020.csv')
    if dados is None:
        return
    
    # Pré-processamento
    dados_proc = preprocessar_dados(dados)
    
    print("\n=== INFORMAÇÕES GERAIS ===")
    print(f"Total de linhas: {dados_proc.shape[0]}")
    print(f"Total de colunas: {dados_proc.shape[1]}")
    
    if 'S' in dados_proc.columns:
        print("\nDistribuição por setor:")
        print(dados_proc['S'].value_counts())
    
    # Parte 1: Regressão Linear
    print("\n" + "="*50)
    print("PARTE 1: REGRESSÃO LINEAR PARA PREVER X44")
    print("="*50)
    
    analise_exploratoria(dados_proc, 'X44')
    X, y = preparar_dados_modelagem(dados_proc, 'X44')
    features_sel = selecionar_features(X, y, n_features=15)
    modelo_reg, coef_reg = treinar_regressao_linear(X[features_sel], y)
    
    # Parte 2: Regressão Logística
    print("\n" + "="*50)
    print("PARTE 2: REGRESSÃO LOGÍSTICA PARA CLASSIFICAÇÃO")
    print("="*50)
    
    var_binaria = criar_variavel_binaria(dados_proc, 'X44')
    
    if 'S' in dados_proc.columns and 3 in dados_proc['S'].unique():
        dados_setor3 = dados_proc[dados_proc['S'] == 3].copy()
        print(f"\nDados do setor 3: {dados_setor3.shape}")
        
        X_s3, y_s3 = preparar_dados_modelagem(dados_setor3, var_binaria)
        features_sel_log = selecionar_features(X_s3, y_s3, n_features=15, modelo=LogisticRegression(max_iter=1000))
        modelo_log = treinar_regressao_logistica(X_s3[features_sel_log], y_s3, setor_filtro=3)
    else:
        print("\nSetor 3 não encontrado nos dados.")

# Executar a análise
if __name__ == "__main__":
    executar_analise()