
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ==================================================
# PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="Vendor Invoice Intelligence Portal",
    page_icon="📦",
    layout="wide"
)

# ==================================================
# FILE PATHS
# ==================================================
FREIGHT_PATH = "/content/predict_freight_model.pkl"
INVOICE_PATH = "/content/invoice_classifier_model.pkl"

SCALER_PATHS = [
    "/content/scaler.pkl",
    "/content/models/scaler.pkl",
    "scaler.pkl",
    "models/scaler.pkl"
]

# ==================================================
# LOAD MODELS
# ==================================================
@st.cache_resource
def load_models():
    if not os.path.exists(FREIGHT_PATH):
        raise FileNotFoundError(f"Model not found: {FREIGHT_PATH}")

    if not os.path.exists(INVOICE_PATH):
        raise FileNotFoundError(f"Model not found: {INVOICE_PATH}")

    freight_data = joblib.load(FREIGHT_PATH)
    invoice_data = joblib.load(INVOICE_PATH)

    freight_model = (
        freight_data["model"]
        if isinstance(freight_data, dict)
        else freight_data
    )

    invoice_model = (
        invoice_data["model"]
        if isinstance(invoice_data, dict)
        else invoice_data
    )

    scaler = None
    scaler_path = None

    for path in SCALER_PATHS:
        if os.path.exists(path):
            scaler = joblib.load(path)
            scaler_path = path
            break

    return (
        freight_data,
        invoice_data,
        freight_model,
        invoice_model,
        scaler,
        scaler_path
    )


try:
    (
        freight_data,
        invoice_data,
        freight_model,
        invoice_model,
        scaler,
        scaler_path
    ) = load_models()

except Exception as e:
    st.error(f"Could not load the saved models: {e}")
    st.stop()


# ==================================================
# SIDEBAR
# ==================================================
st.sidebar.title("🔎 Model Selection")

module = st.sidebar.radio(
    "Choose Prediction Module",
    [
        "Freight Cost Prediction",
        "Invoice Manual Approval Flag"
    ]
)

st.sidebar.divider()
st.sidebar.subheader("Business Impact")
st.sidebar.markdown("""
- 📉 Improved cost forecasting
- 🧾 Identify invoices requiring review
- ⚙️ Support faster finance operations
""")


# ==================================================
# HEADER
# ==================================================
st.title("📦 AI-Driven Freight Cost Prediction & Invoice Risk Flagging")

st.write(
    "Use machine learning to forecast freight costs and identify "
    "vendor invoices that may require manual review."
)

st.divider()


# ==================================================
# MODULE 1: FREIGHT COST PREDICTION
# ==================================================
if module == "Freight Cost Prediction":

    st.header("🚚 Freight Cost Prediction")
    st.write("Predict freight cost using invoice dollars.")

    invoice_dollars = st.number_input(
        "💰 Invoice Dollars ($)",
        min_value=0.0,
        value=18500.0,
        step=100.0
    )

    if st.button("🔮 Predict Freight Cost", use_container_width=True):

        try:
            expected_features = getattr(
                freight_model, "n_features_in_", 1
            )

            if expected_features != 1:
                st.error(
                    f"The saved freight model expects {expected_features} "
                    "features, but this app supplies only Invoice Dollars. "
                    "Check that you loaded the correct saved model."
                )

            else:
                feature_names = getattr(
                    freight_model, "feature_names_in_", None
                )

                # Use the model's original column name, if stored.
                column_name = (
                    feature_names[0]
                    if feature_names is not None
                    else "Invoice Dollars"
                )

                X_freight = pd.DataFrame({
                    column_name: [invoice_dollars]
                })

                prediction = freight_model.predict(X_freight)
                predicted_cost = float(
                    np.asarray(prediction).ravel()[0]
                )

                st.success(
                    f"### Predicted Freight Cost: ${predicted_cost:,.2f}"
                )

                if isinstance(freight_data, dict):
                    col1, col2 = st.columns(2)

                    if "mae" in freight_data:
                        col1.metric(
                            "Model MAE",
                            f"${float(freight_data['mae']):,.2f}"
                        )

                    if "rmse" in freight_data:
                        col2.metric(
                            "Model RMSE",
                            f"${float(freight_data['rmse']):,.2f}"
                        )

        except Exception as e:
            st.error(f"Freight prediction failed: {e}")


#```python
# ==================================================
# MODULE 2: INVOICE MANUAL APPROVAL FLAG
# ==================================================

elif module == "Invoice Manual Approval Flag":

    st.header("🧾 Invoice Manual Approval Flag")
    st.write(
        "Estimate whether a vendor invoice should be flagged "
        "for manual review."
    )

    st.info(
        "This model predicts the saved training label. "
        "A flagged invoice is not necessarily fraudulent."
    )

    # Load the original fitted MinMaxScaler
    scaler = None

    for path in [
        "/content/scaler.pkl",
        "/content/models/scaler.pkl",
        "scaler.pkl",
        "models/scaler.pkl"
    ]:
        if os.path.exists(path):
            scaler = joblib.load(path)
            break

    if scaler is None:
        st.error(
            "scaler.pkl was not found. Place the original fitted "
            "MinMaxScaler file in /content/scaler.pkl."
        )
        st.stop()

    st.success("Original fitted MinMaxScaler loaded.")

    # Input fields
    col1, col2 = st.columns(2)

    with col1:
        invoice_quantity = st.number_input(
            "Invoice Quantity",
            min_value=0.0,
            value=100.0
        )

        invoice_dollars = st.number_input(
            "Invoice Dollars ($)",
            min_value=0.0,
            value=5000.0
        )

        freight = st.number_input(
            "Freight ($)",
            min_value=0.0,
            value=250.0
        )

        days_po_to_invoice = st.number_input(
            "Days PO to Invoice",
            min_value=0.0,
            value=5.0
        )

    with col2:
        total_item_quantity = st.number_input(
            "Total Item Quantity",
            min_value=0.0,
            value=100.0
        )

        total_item_dollars = st.number_input(
            "Total Item Dollars ($)",
            min_value=0.0,
            value=5000.0
        )

        avg_receiving_delay = st.number_input(
            "Average Receiving Delay (Days)",
            min_value=0.0,
            value=3.0
        )

    if st.button("🔎 Check Invoice", use_container_width=True):

        try:
            # EXACT training feature order
            feature_columns = [
                "invoice_quantity",
                "invoice_dollars",
                "Freight",
                "days_po_to_invoice",
                "total_item_quantity",
                "total_item_dollars",
                "avg_receiving_delay"
            ]

            X_input = pd.DataFrame(
                [[
                    invoice_quantity,
                    invoice_dollars,
                    freight,
                    days_po_to_invoice,
                    total_item_quantity,
                    total_item_dollars,
                    avg_receiving_delay
                ]],
                columns=feature_columns
            )

            # Apply the SAME scaler fitted during training
            X_scaled = scaler.transform(X_input)

            # Verify model expects 7 features
            expected = getattr(
                invoice_model, "n_features_in_", 7
            )

            if expected != 7:
                st.error(
                    f"Classifier expects {expected} features, "
                    "but the app supplies 7. Check the saved model."
                )
                st.stop()

            # Predict using the scaled features
            prediction = invoice_model.predict(X_scaled)
            predicted_label = np.asarray(prediction).ravel()[0]

            if predicted_label == 1:
                st.error("⚠️ Invoice flagged for manual review.")
            else:
                st.success(
                    "✅ Invoice not flagged for manual review."
                )

            # Optional prediction probabilities
            if hasattr(invoice_model, "predict_proba"):
                probabilities = invoice_model.predict_proba(X_scaled)[0]
                classes = invoice_model.classes_

                st.subheader("Prediction Confidence")

                for class_label, probability in zip(
                    classes, probabilities
                ):
                    st.write(
                        f"Class {class_label}: {probability:.2%}"
                    )

        except Exception as e:
            st.error(f"Invoice prediction failed: {e}")


# ==================================================
# FOOTER
# ==================================================
st.divider()

st.caption(
    "Vendor Invoice Intelligence Portal | "
    "Machine Learning with Python and Scikit-learn"
)

