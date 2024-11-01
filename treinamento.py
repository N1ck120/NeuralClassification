import pandas as pd  # Importa a biblioteca pandas para manipulação de dados
import numpy as np  # Importa a biblioteca NumPy para manipulação de arrays
import seaborn as sns  # Importa seaborn para visualização de dados
import matplotlib.pyplot as plt  # Importa matplotlib para gráficos
import plotly.express as px  # Importa Plotly para visualizações interativas
from sklearn.model_selection import train_test_split  # Importa função para dividir dados
from sklearn.preprocessing import LabelEncoder, StandardScaler  # Importa classes para pré-processamento
from sklearn.neural_network import MLPClassifier  # Importa a classe para MLP com sklearn
from sklearn.metrics import accuracy_score  # Importa função para calcular a acurácia
import tensorflow as tf  # Importa TensorFlow para construir e treinar a rede neural
from tensorflow.keras.models import Sequential  # Importa o modelo sequencial do Keras
from tensorflow.keras.layers import Dense  # Importa a camada densa do Keras
import joblib  # Importa joblib para salvar e carregar objetos, como o LabelEncoder

# Carregar os dados a partir de um arquivo CSV
base = pd.read_csv('Iris.csv')

# Consultar as 10 primeiras linhas do dataset
print(base.head(10))

# Consultar estatísticas descritivas do dataset
print(base.describe())

# Remover a coluna 'Id', que não é necessária para a análise
base = base.drop('Id', axis=1)

# Remover outliers com base na largura da sépala
base = base[(base['SepalWidthCm'] <= 4) & (base['SepalWidthCm'] >= 2.05)]

# Obter as variáveis previsoras (features) e a saída (target)
X_previsoras = base.iloc[:, 0:4].values  # Seleciona as colunas das características
y_saidas = base.iloc[:, 4].values  # Seleciona a coluna do rótulo (espécie da Iris)

# Padronização das variáveis previsoras
scaler = StandardScaler()
X_previsoras_padroniza = scaler.fit_transform(X_previsoras)  # Ajusta e transforma os dados

# Codificação das saídas usando LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_saidas)  # Codifica as saídas em números

# Salvar o LabelEncoder para uso posterior
joblib.dump(label_encoder, 'label_encoder.pkl')

# Dividir os dados em conjuntos de treino e teste (70% treino, 30% teste)
X_train, X_test, y_train, y_test = train_test_split(X_previsoras_padroniza, y, test_size=0.3, random_state=42)

# Criar a rede neural com sklearn
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 3), activation='relu', solver='adam', max_iter=1000, random_state=42)

# Treinar o modelo
mlp.fit(X_train, y_train)

# Avaliar o modelo no conjunto de treinamento
y_pred_train = mlp.predict(X_train)
accuracy_train = accuracy_score(y_train, y_pred_train)
print(f'Acurácia no conjunto de treinamento: {accuracy_train * 100:.2f}%')

# Avaliar o modelo no conjunto de teste
y_pred_test = mlp.predict(X_test)
accuracy_test = accuracy_score(y_test, y_pred_test)
print(f'Acurácia no conjunto de teste: {accuracy_test * 100:.2f}%')

# Reverter a codificação das previsões para os rótulos originais
y_pred_original = label_encoder.inverse_transform(y_pred_test)
y_test_original = label_encoder.inverse_transform(y_test)

# Exibir previsões e valores reais
print(f'Previsões (decodificadas): {y_pred_original}')
print(f'Reais (decodificados): {y_test_original}')

# Criar a arquitetura da rede neural com TensorFlow
model_RN = Sequential([
    Dense(6, input_shape=(4,), activation='relu'),  # Camada de entrada com 6 neurônios
    Dense(10, activation='relu'),  # Camada oculta com 10 neurônios
    Dense(3, activation='softmax')  # Camada de saída com 3 neurônios (uma para cada espécie)
])

# Compilar o modelo
model_RN.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo
model_RN.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

# Avaliar o modelo no conjunto de treino
loss_train, accuracy_train = model_RN.evaluate(X_train, y_train)
print(f'Acurácia no conjunto de treino: {accuracy_train * 100:.2f}%')

# Avaliar o modelo no conjunto de teste
loss_test, accuracy_test = model_RN.evaluate(X_test, y_test)
print(f'Acurácia no conjunto de teste: {accuracy_test * 100:.2f}%')

# Fazer previsões no conjunto de teste
y_pred = model_RN.predict(X_test)
y_pred_classes = y_pred.argmax(axis=1)  # Obtém a classe com a maior probabilidade

# Exibir previsões e valores reais
y_pred_original = label_encoder.inverse_transform(y_pred_classes)
print(f'Previsões (decodificadas): {y_pred_original}')
print(f'Reais (decodificados): {y_test_original}')

# Salvar o modelo da rede neural para uso posterior
model_RN.save('modelo_treinamento_iris.h5')

# Interação com o usuário para entrada de dados
valor1 = float(input('SepalLengthCm: '))
valor2 = float(input('SepalWidthCm: '))
valor3 = float(input('PetalLengthCm: '))
valor4 = float(input('PetalWidthCm: '))
X_entrada = np.array([[valor1, valor2, valor3, valor4]])  # Cria um array com os dados de entrada

# Abrir o modelo salvo
modelo_aberto = tf.keras.models.load_model('modelo_treinamento_iris.h5')

# Carregar o LabelEncoder salvo
label_encoder = joblib.load('label_encoder.pkl')

# Fazer a predição usando o modelo aberto
y_pred = modelo_aberto.predict(X_entrada)
y_pred_classes = y_pred.argmax(axis=1)  # Obtém a classe prevista

# Reverter a codificação das previsões para os rótulos originais
saida = label_encoder.inverse_transform(y_pred_classes)

# Exibir previsões
print(f'Previsões (decodificadas): {saida}')
