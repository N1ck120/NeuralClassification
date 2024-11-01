from flask import Flask, render_template, request  # Importa as bibliotecas necessárias do Flask
import numpy as np  # Importa NumPy para manipulação de arrays
import tensorflow as tf  # Importa TensorFlow para trabalhar com o modelo
from model import load_model, predict  # Importa funções para carregar o modelo e fazer previsões

app = Flask(__name__)  # Cria uma instância do aplicativo Flask

# Carrega o modelo ao iniciar o aplicativo
model = load_model('modelo_treinamento_iris.h5')

@app.route('/')  # Define a rota para a página inicial
def index():
    return render_template('index.html')  # Renderiza o template HTML da página inicial

@app.route('/predict', methods=['POST'])  # Define a rota para a predição, aceita apenas métodos POST
def predict_view():
    # Obtém os dados de entrada do formulário enviado pelo usuário
    sepal_length = float(request.form['sepal_length'])  # Comprimento da sépala
    sepal_width = float(request.form['sepal_width'])    # Largura da sépala
    petal_length = float(request.form['petal_length'])  # Comprimento da pétala
    petal_width = float(request.form['petal_width'])    # Largura da pétala

    # Faz a predição usando os dados de entrada
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])  # Cria um array NumPy com os dados de entrada
    prediction, accuracy = predict(model, input_data)  # Chama a função predict para obter a previsão e a acurácia
    
    # Renderiza a página inicial novamente, passando os resultados da predição
    return render_template('index.html', prediction=prediction, accuracy=accuracy)

if __name__ == '__main__':
    app.run(debug=True)  # Executa o aplicativo Flask em modo de depuração
