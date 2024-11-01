# treinamento.py

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

def preparar_dados(file_path):
    """
    Carrega e prepara os dados de um arquivo CSV.

    Parâmetros:
    - file_path (str): Caminho para o arquivo CSV.

    Retorna:
    - X (array): Variáveis previsoras padronizadas.
    - y (array): Variável de saída codificada.
    """
    # Carrega os dados
    base = pd.read_csv(file_path)

    # Remove a coluna 'Id' e outliers com base na largura da sépala
    if 'Id' in base.columns:
        base = base.drop('Id', axis=1)
    base = base[(base['SepalWidthCm'] <= 4) & (base['SepalWidthCm'] >= 2.05)]

    # Divide em variáveis previsoras e alvo
    X = base.iloc[:, 0:4].values
    y = base.iloc[:, 4].values

    # Padroniza as variáveis previsoras
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Salva o scaler para uso posterior
    joblib.dump(scaler, 'scaler.pkl')

    # Codifica as saídas
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Salva o codificador para uso posterior
    joblib.dump(label_encoder, 'label_encoder.pkl')

    return X, y

def criar_modelo():
    """
    Cria e retorna a arquitetura da rede neural para classificação.

    Retorna:
    - model (Sequential): Modelo de rede neural.
    """
    model = Sequential([
        Dense(6, input_shape=(4,), activation='relu'),
        Dense(10, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def treinar_modelo(X, y):
    """
    Treina o modelo e salva o melhor modelo.

    Parâmetros:
    - X (array): Variáveis previsoras.
    - y (array): Variável de saída.

    Retorna:
    - accuracy (float): Acurácia do modelo no conjunto de teste.
    """
    # Divide os dados em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Cria o modelo
    model = criar_modelo()

    # Configura o checkpoint para salvar o melhor modelo
    checkpoint = ModelCheckpoint('model.keras', save_best_only=True, monitor='val_accuracy', mode='max')

    # Treina o modelo
    history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), callbacks=[checkpoint])

    # Avalia o modelo nos dados de teste
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Acurácia no conjunto de teste: {accuracy * 100:.2f}%')

    return accuracy

def carregar_modelo():
    """
    Carrega o modelo e os objetos de pré-processamento para predições.

    Retorna:
    - model: Modelo carregado.
    - scaler: Objeto StandardScaler carregado.
    - label_encoder: Objeto LabelEncoder carregado.
    """
    model = tf.keras.models.load_model('model.keras')
    scaler = joblib.load('scaler.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    return model, scaler, label_encoder

def prever_especie(input_data):
    """
    Realiza a predição para uma linha de dados de entrada.

    Parâmetros:
    - input_data (array): Dados de entrada para predição.

    Retorna:
    - species (str): Nome da espécie prevista.
    """
    model, scaler, label_encoder = carregar_modelo()

    # Padroniza os dados de entrada
    input_data = scaler.transform([input_data])

    # Faz a previsão
    y_pred = model.predict(input_data)
    y_pred_class = np.argmax(y_pred, axis=1)

    # Decodifica a classe prevista
    species = label_encoder.inverse_transform(y_pred_class)
    return species[0]
if __name__ == '__main__':
    # Caminho para um arquivo CSV local
    file_path = 'Iris.csv'  # Ajuste o caminho conforme necessário

    # Prepara dados e treina o modelo
    X, y = preparar_dados(file_path)
    treinar_modelo(X, y)

    # Teste de previsão
    sample_input = [5.1, 3.5, 1.4, 0.2]  # Exemplo de dados de entrada
    species = prever_especie(sample_input)
    print(f'A espécie prevista para {sample_input} é: {species}')
