import streamlit as st

from inference.predict_freight import predict_freight_cost
from inference.predict_invoice_flag import predict_invoice_flag

with open("styles.css") as f:
    st.title("📄 Invoice Intelligence Platform")

st.markdown(
    "Machine Learning powered solution for **freight forecasting** and **invoice risk detection**."
)

# -------------------------------------------------
# Short Project Overview
# -------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.info("""
### 📌 About

AI-driven platform that predicts freight costs and identifies risky vendor invoices.
""")

with col2:
    st.info("""
### 🎯 Objectives

• Automate invoice validation

• Forecast freight expenses

• Reduce manual review effort
""")

with col3:
    st.info("""
### 📈 Business Impact

• Improve budgeting

• Detect anomalies early

• Reduce financial losses
""")

st.markdown("---")


# -------------------------------------------------
# Sidebar
# -------------------------------------------------

st.sidebar.title("📊 Invoice Intelligence")

page = st.sidebar.radio(
    "Choose Application",
    [
        "Freight Cost Prediction",
        "Invoice Risk Detection"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    """
    This application uses Machine Learning to:

    - Predict Freight Cost
    - Detect Risky Invoices

    Developed by Akash Das
    """
)




# =================================================
# FREIGHT PREDICTION
# =================================================

if page == "Freight Cost Prediction":

    st.header("🚚 Freight Cost Prediction")

    col1, col2 = st.columns(2)

    with col1:

        quantity = st.number_input(
            "Invoice Quantity",
            min_value=0.0,
            value=100.0
        )

    with col2:

        dollars = st.number_input(
            "Invoice Dollars ($)",
            min_value=0.0,
            value=1000.0
        )

    if st.button("Predict Freight Cost"):

        input_data = {

            'Quantity': quantity,
            'Dollars': dollars

        }

        prediction = predict_freight_cost(input_data)

        st.success(
            f"Predicted Freight Cost: ${prediction:,.2f}"
        )

        st.metric(
            label="Estimated Freight Cost",
            value=f"${prediction:,.2f}"
        )


# =================================================
# INVOICE RISK DETECTION
# =================================================

else:

    st.header("⚠️ Invoice Risk Detection")

    col1, col2 = st.columns(2)

    with col1:

        invoice_quantity = st.number_input(
            "Invoice Quantity",
            min_value=0.0
        )

        invoice_dollars = st.number_input(
            "Invoice Dollars ($)",
            min_value=0.0
        )

        freight = st.number_input(
            "Freight Cost ($)",
            min_value=0.0
        )

    with col2:

        total_item_quantity = st.number_input(
            "Total Item Quantity",
            min_value=0.0
        )

        total_item_dollars = st.number_input(
            "Total Item Dollars ($)",
            min_value=0.0
        )

    if st.button("Detect Invoice Risk"):

        input_data = {

            'invoice_quantity': invoice_quantity,
            'invoice_dollars': invoice_dollars,
            'Freight': freight,
            'total_item_quantity': total_item_quantity,
            'total_item_dollars': total_item_dollars

        }

        prediction = predict_invoice_flag(input_data)

        if prediction == 1:

            st.error("⚠️ High Risk Invoice Detected")

            st.metric(
                label="Risk Status",
                value="HIGH RISK"
            )

        else:

            st.success("✅ Invoice Appears Normal")

            st.metric(
                label="Risk Status",
                value="NORMAL"
            )
