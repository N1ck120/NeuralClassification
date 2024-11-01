# app.py

from flask import Flask, render_template, request, redirect, url_for
from treinamento import preparar_dados, treinar_modelo, prever_especie
import os

app = Flask(__name__)

# Configuração do diretório de upload
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' in request.files:
            # Parte de treinamento
            file = request.files['file']
            if file.filename != '':
                file_path = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(file_path)
                X, y = preparar_dados(file_path)
                accuracy = treinar_modelo(X, y)
                return render_template('index.html', accuracy=accuracy)
        else:
            # Parte de predição
            sepal_length = float(request.form['sepal_length'])
            sepal_width = float(request.form['sepal_width'])
            petal_length = float(request.form['petal_length'])
            petal_width = float(request.form['petal_width'])
            input_data = [sepal_length, sepal_width, petal_length, petal_width]
            species = prever_especie(input_data)
            return render_template('index.html', species=species)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
