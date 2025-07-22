# app.py
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
with open("iris_model.pkl", "rb") as f:
    model, scaler = pickle.load(f)

# Iris class info
iris_info = {
    0: ("Setosa", "Setosa is the smallest iris flower with short petals and sepals."),
    1: ("Versicolor", "Versicolor has medium-sized petals and typically purple-blue flowers."),
    2: ("Virginica", "Virginica is the largest with long, wide petals and sepals.")
}

@app.route("/", methods=["GET", "POST"])
def predict():
    result = ""
    if request.method == "POST":
        try:
            # Collect input
            features = [
                float(request.form['sepal_length']),
                float(request.form['sepal_width']),
                float(request.form['petal_length']),
                float(request.form['petal_width'])
            ]

            # Scale the input
            features_scaled = scaler.transform([features])

            # Predict
            prediction = model.predict(features_scaled)[0]
            name, desc = iris_info[prediction]
            result = f"<h2>Predicted Flower: {name}</h2><p>{desc}</p>"
        except Exception as e:
            result = f"<p style='color:red;'>Error: {e}</p>"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

