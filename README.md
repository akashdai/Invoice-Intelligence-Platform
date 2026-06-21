# 📄 Invoice Intelligence Platform

A machine learning–powered web application that predicts **freight costs** and detects **high-risk vendor invoices**, built to help finance and logistics teams catch anomalies early and forecast shipping expenses with greater accuracy.
<img width="1917" height="971" alt="Image" src="https://github.com/user-attachments/assets/a853865d-98ec-49bc-adcf-12df673bfae8" />

🔗 **Live Demo:** _[https://invoice-intelligence-platform-2ufznsnzemmndefgyddqjw.streamlit.app/]_

---

## 🎯 Overview

Manually reviewing invoices and estimating freight costs is slow and error-prone. This platform automates both tasks using trained ML models, wrapped in a simple, interactive Streamlit interface — so users can get instant predictions without touching code.

**Core capabilities:**
- **Freight Cost Prediction** — Estimates expected freight cost from invoice quantity and dollar value using a trained regression model.
- **Invoice Risk Detection** — Flags potentially risky or anomalous invoices based on quantity, dollar amount, and freight cost patterns, using a trained classification model.

---
# 📸 Application Screenshots

## 🚚 Freight Cost Prediction

<img width="1917" height="971" alt="Image" src="https://github.com/user-attachments/assets/a853865d-98ec-49bc-adcf-12df673bfae8" />

The freight prediction module estimates shipping costs instantly based on invoice quantity and invoice dollar value.

---

## ⚠️ Invoice Risk Detection

![Invoice Risk Detection](Images/invoice-risk-detection.png)

The invoice risk detection module identifies potentially risky invoices and flags anomalies for further review.

---


## 🧠 How It Works

1. Historical invoice and freight data is cleaned and preprocessed (`data_preprocessing.py`).
2. Two separate models are trained and evaluated:
   - A **regression model** for freight cost forecasting (`freight_cost_prediction/`)
   - A **classification model** for invoice risk flagging (`invoice flagging/`)
3. The best-performing models are serialized with `joblib` and saved to `models/`.
4. The Streamlit app (`app.py`) loads these models at runtime and serves predictions through a simple form-based UI (`inference/`).

---

## 🗂️ Project Structure
Invoice-Intelligence-Platform/

├── app.py                          # Streamlit app entry point

├── inference/                      # Inference logic for loading models & predicting

│   ├── predict_freight.py

│   └── predict_invoice_flag.py

├── frieght_cost_prediction/        # Freight cost regression model: preprocessing, training, evaluation

├── invoice flagging/                # Invoice risk classification model: preprocessing, training, evaluation

├── models/                         # Trained model artifacts (.pkl)

├── notebooks/                      # Exploratory analysis & experimentation notebooks

├── styles.css                      # Custom UI styling

└── requirements.txt                # Python dependencies

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit** — web app framework
- **scikit-learn** — model training & inference
- **pandas / numpy** — data processing
- **joblib** — model serialization
- **matplotlib / seaborn** — exploratory visualization

---

## 🚀 Getting Started

### Run locally

```bash
git clone https://github.com/akashdai/Invoice-Intelligence-Platform.git
cd Invoice-Intelligence-Platform
pip install -r requirements.txt
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 📊 Using the App

**Freight Cost Prediction**
Enter invoice quantity and dollar amount → get an estimated freight cost.

**Invoice Risk Detection**
Enter invoice quantity, dollar value, freight cost, and item totals → the model flags the invoice as **Normal** or **High Risk**.

---

## 📈 Model Development

Model training and evaluation notebooks are available under `notebooks/`, covering data exploration, feature engineering, and comparison of candidate models before selecting the final versions saved in `models/`.

---

## 👤 Author

**Akash Das**

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
