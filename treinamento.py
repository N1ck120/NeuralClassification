import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Carregar os dados
base = pd.read_csv('Iris.csv')

# Consultar as 10 primeiras linhas
print(base.head(10))

# Consultar estatísticas descritivas
print(base.describe())

# Remover a coluna 'Id'
base = base.drop('Id', axis=1)

# Remover outliers
base = base[(base['SepalWidthCm'] <= 4) & (base['SepalWidthCm'] >= 2.05)]

# Obter as variáveis previsoras e a saída
X_previsoras = base.iloc[:, 0:4].values
y_saidas = base.iloc[:, 4].values  # Ajuste para uma série 1D

# Padronização
scaler = StandardScaler()
X_previsoras_padroniza = scaler.fit_transform(X_previsoras)

# Codificação das saídas
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_saidas)

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_previsoras_padroniza, y, test_size=0.3, random_state=42)

# Rede Neural com sklearn
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

# Reverter a codificação das previsões
y_pred_original = label_encoder.inverse_transform(y_pred_test)
y_test_original = label_encoder.inverse_transform(y_test)

# Exibir previsões e valores reais
print(f'Previsões (decodificadas): {y_pred_original}')
print(f'Reais (decodificados): {y_test_original}')

# Criar a arquitetura da rede neural com TensorFlow
model_RN = Sequential([
    Dense(6, input_shape=(4,), activation='relu'),
    Dense(10, activation='relu'),
    Dense(3, activation='softmax')
])

# Compilar o modelo
model_RN.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo
model_RN.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

# Avaliar o modelo
loss_train, accuracy_train = model_RN.evaluate(X_train, y_train)
print(f'Acurácia no conjunto de treino: {accuracy_train * 100:.2f}%')

# Avaliar o modelo no conjunto de teste
loss_test, accuracy_test = model_RN.evaluate(X_test, y_test)
print(f'Acurácia no conjunto de teste: {accuracy_test * 100:.2f}%')

# Fazer previsões
y_pred = model_RN.predict(X_test)
y_pred_classes = y_pred.argmax(axis=1)

# Exibir previsões e valores reais
y_pred_original = label_encoder.inverse_transform(y_pred_classes)
print(f'Previsões (decodificadas): {y_pred_original}')
print(f'Reais (decodificados): {y_test_original}')

# Salvar o modelo da Rede Neural
model_RN.save('modelo_treinamento_iris.h5')

# Interação com o usuário para entrada de dados
valor1 = float(input('SepalLengthCm: '))
valor2 = float(input('SepalWidthCm: '))
valor3 = float(input('PetalLengthCm: '))
valor4 = float(input('PetalWidthCm: '))
X_entrada = np.array([[valor1, valor2, valor3, valor4]])

# Abrir o modelo salvo
modelo_aberto = tf.keras.models.load_model('/content/drive/MyDrive/dados/modelo_RN.h5')

# Fazer a predição
y_pred = modelo_aberto.predict(X_entrada)
y_pred_classes = y_pred.argmax(axis=1)

# Reverter a codificação das previsões
saida = label_encoder.inverse_transform(y_pred_classes)

# Exibir previsões
print(f'Previsões (decodificadas): {saida}')
