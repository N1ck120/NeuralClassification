from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np

app = Flask(__name__)

scaler = joblib.load("models/scaler.pkl")
model = joblib.load("models/model.pkl")

@app.route('/', methods=['GET', 'POST'])
def index():
    # Parte de predição
    predicted_fruit = None
    if request.method == "POST":
        fruit_weight = float(request.form['fruit_weight'])
        fruit_size = float(request.form['fruit_size'])
        entry = np.array([[fruit_weight, fruit_size]])
        standard_entry = scaler.transform(entry)
        pred = model.predict(standard_entry)
        predicted = "Maçã" if pred[0] == 0 else "Laranja"
        return render_template('index.html', predicted_fruit = predicted)
    else :
        return render_template('index.html', predicted_fruit = "Nada")
if __name__ == '__main__':
    app.run(debug=True)
