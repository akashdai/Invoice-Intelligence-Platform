# predict_invoice_flag.py

import os
import joblib
import pandas as pd


# Absolute model path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_invoice_risk_model.pkl"
)


def load_model():

    model = joblib.load(MODEL_PATH)

    return model


def predict_invoice_flag(input_data):

    model = load_model()

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)

    return prediction[0]


# Example Inference Run (Local Testing)

if __name__ == "__main__":

    sample_invoice = {

        'invoice_quantity': 500,
        'invoice_dollars': 12000,
        'Freight': 800,
        'total_item_quantity': 520,
        'total_item_dollars': 11800

    }

    prediction = predict_invoice_flag(
        sample_invoice
    )

    if prediction == 1:
        print("⚠️ Risky Invoice")

    else:
        print("✅ Normal Invoice")
    