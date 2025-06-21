from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model
model = joblib.load('xgb_model.joblib')

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)  # Get data as JSON
    input_df = pd.DataFrame([data])      # Convert to DataFrame

    # Predict
    prediction = model.predict(input_df)[0]

    return jsonify({'predicted_price': round(prediction, 2)})

# Optional root route
@app.route('/')
def home():
    return "Car Price Prediction API is running!"

if __name__ == '__main__':
    app.run(debug=True)
