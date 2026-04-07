from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model safely
if not os.path.exists("model1.pkl"):
    raise Exception("Model not found. Run training first.")

model = joblib.load("model1.pkl")

@app.route('/')
def home():
    return "ML Model API is running"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['input']
    data = np.array(data).reshape(1, -1)
    prediction = model.predict(data)

    return jsonify({
        "prediction": prediction.tolist()
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)