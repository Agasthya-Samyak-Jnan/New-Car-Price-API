from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os
from car_model import CarPriceModel

app = Flask(__name__)
CORS(app)  # Allow CORS from any origin

# Load model (must be a saved CarPriceModel object)
model = joblib.load('car_price_model_with_preprocessing.joblib')

# -------------------------------
# Routes
# -------------------------------
@app.route('/')
def home():
    return "Car Price Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    try:
        prediction = model.predict(data)
        return jsonify({'predicted_price': round(float(prediction), 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# -------------------------------
# Required for Render
# -------------------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
