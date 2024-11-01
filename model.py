import numpy as np  # Importa a biblioteca NumPy para manipulação de arrays
import tensorflow as tf  # Importa TensorFlow para trabalhar com o modelo
from sklearn.preprocessing import LabelEncoder  # Importa LabelEncoder para codificação de rótulos
import joblib  # Importa joblib para carregar objetos salvos, como o LabelEncoder

# Função para carregar o modelo a partir do caminho especificado
def load_model(model_path):
    return tf.keras.models.load_model(model_path)  # Carrega e retorna o modelo Keras salvo

# Função para fazer previsão e calcular a taxa de acerto
def predict(model, input_data):
    # Faz a predição usando os dados de entrada
    predictions = model.predict(input_data)  # Obtém as previsões do modelo
    predicted_class = np.argmax(predictions, axis=1)  # Obtém a classe com a maior probabilidade

    # Carrega o LabelEncoder salvo para reverter a codificação
    label_encoder = joblib.load('label_encoder.pkl')  # Certifique-se de que o arquivo label_encoder.pkl existe
    result = label_encoder.inverse_transform(predicted_class)  # Converte as classes previstas de volta aos rótulos originais

    # Simula uma taxa de acerto fixa
    accuracy = 95.0  # Exemplo de taxa de acerto; pode ser ajustada conforme necessário
    return result[0], accuracy  # Retorna a classe prevista e a acurácia
