import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# ===============================
# 1. Carregar e concatenar os ficheiros CSV
# ===============================
pasta = os.path.dirname(os.path.abspath(__file__))
lista_df = []

for nome_ficheiro in os.listdir(pasta):
    if nome_ficheiro.endswith('.csv') and nome_ficheiro[:4].isdigit():
        caminho = os.path.join(pasta, nome_ficheiro)
        try:
            df_temp = pd.read_csv(caminho, na_values='m')
            df_temp['Ano'] = int(nome_ficheiro[:4])
            lista_df.append(df_temp)
            print(f"✅ {nome_ficheiro} carregado com sucesso.")
        except Exception as e:
            print(f"❌ Erro ao ler {nome_ficheiro}: {e}")

df = pd.concat(lista_df, ignore_index=True)
print(f"\n📦 Dados combinados: {df.shape[0]} linhas, {df.shape[1]} colunas")

# ===============================
# 2. Pré-processamento
# ===============================
df = df.drop(columns=['X1'], errors='ignore')
df = df.dropna()

if df['S'].dtype == object:
    df['S'] = LabelEncoder().fit_transform(df['S'])

X = df.drop(columns=['S', 'Country', 'Ano', 'Num'], errors='ignore')
y = df['S']

# ===============================
# 3. Dividir em treino e teste
# ===============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ===============================
# 4. Treinar o modelo (com profundidade reduzida)
# ===============================
clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X_train, y_train)

# ===============================
# 5. Avaliar o desempenho
# ===============================
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Acurácia do modelo: {acc * 100:.2f}%")

mat_conf = confusion_matrix(y_test, y_pred)
print("📊 Matriz de Confusão:")
print(mat_conf)

# ===============================
# 6. Gerar e guardar a árvore de decisão
# ===============================
plt.figure(figsize=(40, 25))
plot_tree(clf, feature_names=X.columns, class_names=True, filled=True, fontsize=18)
plt.tight_layout()
plt.savefig("arvore_decisao.png")
plt.show()
