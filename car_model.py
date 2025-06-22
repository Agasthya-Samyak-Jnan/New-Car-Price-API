import pandas as pd

class CarPriceModel:
    def __init__(self, model, mappings):
        self.model = model
        self.mappings = mappings

    def safe_lower(self, val):
        return val.lower() if isinstance(val, str) else val

    def preprocess(self, raw_input):
        df = pd.DataFrame([{
            "Levy": float(raw_input['levy']),
            "Manufacturer": self.mappings['manufacturer'].get(self.safe_lower(raw_input['manufacturer']), 0),
            "Category": self.mappings['category'].get(self.safe_lower(raw_input['category']), 0),
            "Leather_interior": self.mappings['leather'][self.safe_lower(raw_input['leather'])] if isinstance(raw_input['leather'], str) else int(raw_input['leather']),
            "Engine_volume": float(raw_input['engine_volume']),
            "Mileage": int(raw_input['mileage']),
            "Cylinders": int(raw_input['cylinders']),
            "Wheel": self.mappings['wheel'][self.safe_lower(raw_input['wheel'])],
            "Airbags": int(raw_input['airbags']),
            "Turbo": int(raw_input['turbo']),
            "Drive_4x4": self.safe_lower(raw_input['drive']) == "4x4",
            "Drive_front": self.safe_lower(raw_input['drive']) == "front",
            "Drive_rear": self.safe_lower(raw_input['drive']) == "rear",
            "Gear_automatic": self.safe_lower(raw_input['gearbox']) == "automatic",
            "Gear_manual": self.safe_lower(raw_input['gearbox']) == "manual",
            "Gear_tiptronic": self.safe_lower(raw_input['gearbox']) == "tiptronic",
            "Gear_variator": self.safe_lower(raw_input['gearbox']) == "variator",
            "Fuel_cng": self.safe_lower(raw_input['fuel']) == "cng",
            "Fuel_diesel": self.safe_lower(raw_input['fuel']) == "diesel",
            "Fuel_hybrid": self.safe_lower(raw_input['fuel']) == "hybrid",
            "Fuel_hydrogen": self.safe_lower(raw_input['fuel']) == "hydrogen",
            "Fuel_lpg": self.safe_lower(raw_input['fuel']) == "lpg",
            "Fuel_petrol": self.safe_lower(raw_input['fuel']) == "petrol",
            "Fuel_plug-in hybrid": self.safe_lower(raw_input['fuel']) == "plug-in hybrid"
        }])
        return df

    def predict(self, raw_input):
        processed = self.preprocess(raw_input)
        prediction = self.model.predict(processed)
        return prediction[0]
