import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from kneed import KneeLocator
import os

# Configuração para melhor visualização dos gráficos
plt.style.use('seaborn-v0_8-whitegrid')
sns.set(font_scale=1.2)
colors = sns.color_palette("viridis", 6)

# Função para carregar e tratar os dados dos arquivos CSV
def load_data(years=[2017, 2018, 2019, 2020]):
    all_data = []
    
    for year in years:
        filename = f"{year}.csv"
        try:
            # Tentativa de leitura do arquivo CSV
            df = pd.read_csv(filename, na_values=['m'])
            df['Year'] = year  # Adicionar coluna para identificar o ano
            all_data.append(df)
            print(f"  • {filename}: ✅ Carregado com sucesso")
        except FileNotFoundError:
            print(f"  • {filename}: ❌ Arquivo não encontrado")
    
    if not all_data:
        print("❌ ERRO: Nenhum arquivo foi carregado.")
        return None
    
    # Combinando todos os DataFrames
    combined_data = pd.concat(all_data, ignore_index=True)
    print(f"\n✅ Dataset combinado: {combined_data.shape[0]} empresas × {combined_data.shape[1]} variáveis")
    
    return combined_data

# Função para pré-processamento dos dados
def preprocess_data(df):
    if df is None:
        return None, None
    
    # Exibindo informações sobre valores faltantes
    missing_values = df.isnull().sum().sum()
    print(f"  • Valores faltantes encontrados: {missing_values}")
    
    # Extraindo colunas categóricas e numéricas
    categorical_cols = ['Country', 'S', 'Year']
    numeric_cols = [col for col in df.columns if col.startswith('X')]
    
    # Verificando se todas as colunas numéricas estão presentes
    print(f"  • Indicadores econômicos encontrados: {len(numeric_cols)}")
    
    # Criando um novo DataFrame contendo apenas as colunas relevantes
    data_processed = df[categorical_cols + numeric_cols].copy()
    
    # Lidando com valores faltantes nas colunas numéricas (substituir por média)
    for col in numeric_cols:
        data_processed[col] = data_processed[col].fillna(data_processed[col].mean())
    
    # Detecção e tratamento de outliers
    print("\n🔍 DETECÇÃO E TRATAMENTO DE OUTLIERS:")
    
    # Criando cópias das colunas numéricas para análise
    X = data_processed[numeric_cols].copy()
    
    # Calculando Z-scores para cada coluna numérica
    z_scores = pd.DataFrame()
    for col in numeric_cols:
        z_scores[col] = (X[col] - X[col].mean()) / X[col].std()
    
    # Identificando outliers usando Z-score (|z| > 3)
    outliers_mask = (z_scores.abs() > 3).any(axis=1)
    outliers_count = outliers_mask.sum()
    print(f"  • Outliers identificados: {outliers_count} registros ({outliers_count/len(data_processed)*100:.1f}% do total)")
    
    if outliers_count > 0:
        # Mostrando distribuição de outliers por país
        outliers_by_country = data_processed.loc[outliers_mask, 'Country'].value_counts()
        print(f"\n  • Distribuição de outliers por país:")
        for country, count in outliers_by_country.items():
            print(f"      - {country}: {count} registros")
        
        # Mostrando distribuição de outliers por setor
        sector_map = {
            1: 'Transporte', 
            2: 'Comércio Atacadista', 
            3: 'Manufatura',
            4: 'Varejo', 
            5: 'Energia', 
            6: 'Construção Civil'
        }
        outliers_by_sector = data_processed.loc[outliers_mask, 'S'].value_counts()
        print(f"\n  • Distribuição de outliers por setor:")
        for sector, count in outliers_by_sector.items():
            print(f"      - {sector_map[sector]}: {count} registros")
        
        # Removendo outliers
        print(f"\n  • Removendo outliers para melhorar qualidade do clustering...")
        data_processed = data_processed.loc[~outliers_mask]
        print(f"  • Dataset após remoção: {len(data_processed)} empresas × {len(data_processed.columns)} variáveis")
    else:
        print("  • Não foram encontrados outliers significativos.")
    
    # Contando valores únicos nas variáveis categóricas
    print("\nAnalisando dados categóricos:")
    print(f"  • Países: {', '.join(data_processed['Country'].unique())}")
    
    sector_map = {
        1: 'Transporte', 
        2: 'Comércio Atacadista', 
        3: 'Manufatura',
        4: 'Varejo', 
        5: 'Energia', 
        6: 'Construção Civil'
    }
    sectors = [f"{s} ({sector_map[s]})" for s in sorted(data_processed['S'].unique())]
    print(f"  • Setores: {', '.join(sectors)}")
    
    # Exibindo informações sobre o dataset processado
    print(f"\n✅ Dataset processado e pronto para análise!")
    
    # Retornando o dataset processado e as colunas numéricas
    return data_processed, numeric_cols

# Função para determinar o número ideal de clusters (método do cotovelo)
def find_optimal_clusters(data, numeric_cols, max_clusters=11):
    # Preparação dos dados para clustering
    X = data[numeric_cols].values
    
    # Padronizando os dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Calculando inércia para diferentes valores de k
    print("\nCalculando inércia para valores de k de 1 a 10...")
    inertia_values = []
    k_values = range(1, max_clusters)
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertia_values.append(kmeans.inertia_)
        print(f"  • k={k}: inércia = {kmeans.inertia_:.2f}")
    
    # Plotando o gráfico do método do cotovelo
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, inertia_values, 'o-', color='blue', linewidth=2, markersize=8)
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Inércia')
    plt.title('Método do Cotovelo para Determinar o Número Ideal de Clusters')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('elbow_method.png')
    plt.close()
    
    # Usando a biblioteca KneeLocator para encontrar o ponto ótimo
    try:
        kneedle = KneeLocator(k_values, inertia_values, 
                           curve='convex', 
                           direction='decreasing')
        optimal_k = kneedle.elbow
        print(f"\n✅ Método do cotovelo identificou {optimal_k} como número ideal de clusters")
    except:
        # Se ocorrer algum erro com o KneeLocator, definimos um valor padrão
        optimal_k = 4
        print(f"\n⚠️ Ocorreu um erro ao usar KneeLocator. Usando valor padrão: {optimal_k} clusters")
    
    return optimal_k, X_scaled

# Função para realizar clustering e visualizar resultados
def perform_clustering(data, numeric_cols, n_clusters, X_scaled=None):
    if X_scaled is None:
        # Preparação dos dados para clustering (caso não tenha sido feito antes)
        X = data[numeric_cols].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    
    # Aplicando K-means com o número ideal de clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Adicionando os rótulos de cluster ao DataFrame
    data['Cluster'] = cluster_labels
    
    # Executando PCA para visualização em 2D
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)
    
    # Criando um DataFrame com os componentes principais
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = cluster_labels
    pca_df['Country'] = data['Country'].values
    pca_df['Sector'] = data['S'].values
    pca_df['Year'] = data['Year'].values
    
    # Mapeando valores de setor para nomes descritivos
    sector_names = {
        1: 'Transporte',
        2: 'Comércio Atacadista',
        3: 'Manufatura',
        4: 'Varejo',
        5: 'Energia',
        6: 'Construção Civil'
    }
    pca_df['Sector_Name'] = pca_df['Sector'].map(sector_names)
    
    # Visualização dos clusters usando PCA
    plt.figure(figsize=(12, 8))
    for cluster in range(n_clusters):
        plt.scatter(
            pca_df[pca_df['Cluster'] == cluster]['PC1'], 
            pca_df[pca_df['Cluster'] == cluster]['PC2'], 
            s=50, 
            label=f'Cluster {cluster}'
        )
    
    plt.title(f'Visualização dos {n_clusters} Clusters usando PCA')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('clusters_pca.png')
    plt.close()
    
    # Visualização dos clusters por país
    plt.figure(figsize=(12, 8))
    countries = pca_df['Country'].unique()
    for i, country in enumerate(countries):
        country_data = pca_df[pca_df['Country'] == country]
        plt.scatter(
            country_data['PC1'], 
            country_data['PC2'], 
            s=50,
            label=country
        )
    
    plt.title('Visualização dos Dados por País (PCA)')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('clusters_by_country.png')
    plt.close()
    
    # Visualização dos clusters por setor
    plt.figure(figsize=(12, 8))
    for i, (sector_id, sector_name) in enumerate(sector_names.items()):
        sector_data = pca_df[pca_df['Sector'] == sector_id]
        if len(sector_data) > 0:  # Verifica se existem dados para este setor
            plt.scatter(
                sector_data['PC1'], 
                sector_data['PC2'], 
                s=50,
                label=sector_name
            )
    
    plt.title('Visualização dos Dados por Setor (PCA)')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('clusters_by_sector.png')
    plt.close()
    
    # Análise da distribuição de países e setores por cluster
    print("\n" + "="*80)
    print("🔍 ANÁLISE DA DISTRIBUIÇÃO DE PAÍSES E SETORES POR CLUSTER")
    print("="*80)
    
    # Mapeamento de setores para nomes descritivos
    sector_names = {
        1: 'Transporte',
        2: 'Comércio Atacadista',
        3: 'Manufatura',
        4: 'Varejo',
        5: 'Energia',
        6: 'Construção Civil'
    }
    
    for i in range(n_clusters):
        cluster_data = data[data['Cluster'] == i]
        cluster_size = len(cluster_data)
        cluster_percentage = (cluster_size / len(data)) * 100
        
        print(f"\n📊 CLUSTER {i} ({cluster_size} empresas - {cluster_percentage:.1f}% do total):")
        
        # Distribuição por país
        print("\n   📍 Distribuição por País:")
        country_dist = cluster_data['Country'].value_counts(normalize=True) * 100
        for country, percentage in country_dist.items():
            print(f"      • {country}: {percentage:.1f}%")
        
        # Distribuição por setor
        print("\n   🏭 Distribuição por Setor:")
        sector_dist = cluster_data['S'].value_counts(normalize=True) * 100
        for sector, percentage in sector_dist.items():
            sector_name = sector_names.get(sector, f"Setor {sector}")
            print(f"      • {sector_name}: {percentage:.1f}%")
    
    # Análise dos centroides dos clusters
    print("\n" + "="*80)
    print("🔬 CARACTERÍSTICAS MAIS RELEVANTES POR CLUSTER")
    print("="*80)
    
    cluster_centers = pd.DataFrame(
        kmeans.cluster_centers_,
        columns=numeric_cols
    )
    
    for i in range(n_clusters):
        # Ordenando os indicadores pelo valor absoluto do centroide
        sorted_features = cluster_centers.iloc[i].abs().sort_values(ascending=False)
        top_features = sorted_features.head(5)
        
        print(f"\n⭐ CLUSTER {i} - Top 5 indicadores com maior influência:")
        for feature, value in top_features.items():
            print(f"   • {feature}: {value:.4f}")

    
    return data, cluster_centers

# Função principal
def main():
    print("\n" + "="*80)
    print("🚀 ANÁLISE DE CLUSTERING K-MEANS - EMPRESAS DO GRUPO VISEGRAD (2017-2020)")
    print("="*80)
    
    # Carregando dados
    print("\n📂 CARREGANDO DATASETS...")
    data = load_data()
    
    # Pré-processamento
    print("\n🔧 PRÉ-PROCESSANDO OS DADOS...")
    processed_data, numeric_cols = preprocess_data(data)
    
    if processed_data is None:
        print("❌ Não foi possível processar os dados. Verificar os arquivos de entrada.")
        return
    
    # Determinando o número ideal de clusters
    print("\n🔍 DETERMINANDO O NÚMERO IDEAL DE CLUSTERS...")
    optimal_k, X_scaled = find_optimal_clusters(processed_data, numeric_cols)
    print(f"\n✅ Número ideal de clusters encontrado: {optimal_k}")
    
    # Realizando o clustering com o número ideal de clusters
    print(f"\n📊 APLICANDO K-MEANS COM {optimal_k} CLUSTERS...")
    clustered_data, cluster_centers = perform_clustering(processed_data, numeric_cols, optimal_k, X_scaled)
    
    # Analisando os resultados para diferentes anos
    print("\n" + "="*80)
    print("📆 EVOLUÇÃO DOS CLUSTERS AO LONGO DOS ANOS")
    print("="*80)
    
    for year in sorted(clustered_data['Year'].unique()):
        year_data = clustered_data[clustered_data['Year'] == year]
        cluster_counts = year_data['Cluster'].value_counts(normalize=True) * 100
        
        print(f"\n📅 ANO {year}:")
        for cluster, percentage in cluster_counts.items():
            print(f"   • Cluster {cluster}: {percentage:.1f}% das empresas")
    
    # Análise de evolução temporal (se existirem múltiplos anos)
    if len(clustered_data['Year'].unique()) > 1:
        cluster_evolution = pd.crosstab(
            clustered_data['Year'], 
            clustered_data['Cluster'], 
            normalize='index'
        )
        
        plt.figure(figsize=(12, 8))
        cluster_evolution.plot(kind='bar', stacked=True, colormap='viridis')
        plt.title('Evolução da Distribuição de Clusters ao Longo dos Anos')
        plt.xlabel('Ano')
        plt.ylabel('Proporção')
        plt.legend(title='Cluster')
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig('cluster_evolution.png')
        plt.close()
    
    print("\n" + "="*80)
    print("✅ ANÁLISE DE CLUSTERING CONCLUÍDA COM SUCESSO!")
    print("="*80)
    print("\nArquivos gerados:")
    print("  • elbow_method.png - Gráfico do método do cotovelo")
    print("  • clusters_pca.png - Visualização dos clusters usando PCA")
    print("  • clusters_by_country.png - Distribuição dos dados por país")
    print("  • clusters_by_sector.png - Distribuição dos dados por setor")
    print("  • cluster_evolution.png - Evolução dos clusters ao longo do tempo")
    print("="*80)

if __name__ == "__main__":
    main()