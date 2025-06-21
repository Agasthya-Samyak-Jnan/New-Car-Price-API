from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Custom wrapper class
class CarPriceModel:
    def __init__(self, model, mappings):
        self.model = model
        self.mappings = mappings

    def preprocess(self, raw_input):
        df = pd.DataFrame([{
            "Levy": float(raw_input['levy']),
            "Manufacturer": self.mappings['manufacturer'].get(raw_input['manufacturer'].lower(), 0),
            "Category": self.mappings['category'].get(raw_input['category'].lower(), 0),
            "Leather_interior": self.mappings['leather'][raw_input['leather'].lower()],
            "Engine_volume": float(raw_input['engine_volume']),
            "Mileage": int(raw_input['mileage']),
            "Cylinders": int(raw_input['cylinders']),
            "Wheel": self.mappings['wheel'][raw_input['wheel'].lower()],
            "Airbags": int(raw_input['airbags']),
            "Turbo": int(raw_input['turbo']),
            "Drive_4x4": raw_input['drive'] == "4x4",
            "Drive_front": raw_input['drive'] == "front",
            "Drive_rear": raw_input['drive'] == "rear",
            "Gear_automatic": raw_input['gearbox'] == "automatic",
            "Gear_manual": raw_input['gearbox'] == "manual",
            "Gear_tiptronic": raw_input['gearbox'] == "tiptronic",
            "Gear_variator": raw_input['gearbox'] == "variator",
            "Fuel_cng": raw_input['fuel'] == "cng",
            "Fuel_diesel": raw_input['fuel'] == "diesel",
            "Fuel_hybrid": raw_input['fuel'] == "hybrid",
            "Fuel_hydrogen": raw_input['fuel'] == "hydrogen",
            "Fuel_lpg": raw_input['fuel'] == "lpg",
            "Fuel_petrol": raw_input['fuel'] == "petrol",
            "Fuel_plug-in hybrid": raw_input['fuel'] == "plug-in hybrid"
        }])
        return df

    def predict(self, raw_input):
        processed = self.preprocess(raw_input)
        prediction = self.model.predict(processed)
        return prediction[0]

# App
app = Flask(__name__)

# Load model
model = joblib.load('car_price_model_with_preprocessing.joblib')

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
