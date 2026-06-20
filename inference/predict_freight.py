# predict_freight.py

import os
import joblib
import pandas as pd


# Absolute model path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_freight_model.pkl"
)


def load_model():

    model = joblib.load(MODEL_PATH)

    return model


def predict_freight_cost(input_data):

    model = load_model()

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)

    return prediction[0]


# Example Inference Run (Local Testing)

if __name__ == "__main__":

    sample_invoice = {

        'Quantity': 500,
        'Dollars': 10000

    }

    predicted_freight = predict_freight_cost(
        sample_invoice
    )

    print(f"Predicted Freight Cost: {predicted_freight:.2f}")