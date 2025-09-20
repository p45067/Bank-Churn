import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------- Load artifacts ----------
@st.cache_resource
def load_artifacts():
    model = joblib.load("bank_churn_model.pkl")
    scaler = joblib.load("scaler.pkl")
    with open("feature_columns.json", "r") as f:
        feature_cols = json.load(f)
    return model, scaler, feature_cols

model, scaler, feature_cols = load_artifacts()

st.set_page_config(page_title="Bank Churn Predictor", page_icon="🏦", layout="centered")
st.title("🏦 Bank Customer Churn Predictor")
st.write("Enter customer details to estimate churn probability.")

# ---------- Input form ----------
with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.number_input("Credit Score", min_value=0, max_value=1000, value=650, step=1)
        country = st.selectbox("Country", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Female", "Male"])
        age = st.number_input("Age", min_value=18, max_value=120, value=40, step=1)
        tenure = st.number_input("Tenure (years with bank)", min_value=0, max_value=50, value=5, step=1)

    with col2:
        balance = st.number_input("Balance", min_value=0.0, value=50000.0, step=100.0, format="%.2f")
        products_number = st.number_input("Number of Products", min_value=1, max_value=10, value=2, step=1)
        credit_card = st.selectbox("Has Credit Card?", ["No", "Yes"])
        active_member = st.selectbox("Active Member?", ["No", "Yes"])
        estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=100000.0, step=100.0, format="%.2f")

    threshold = st.slider("Decision Threshold (churn if prob ≥ threshold)", 0.0, 1.0, 0.5, 0.01)

    submitted = st.form_submit_button("Predict")

# ---------- Preprocessing helper ----------
def preprocess_single_row(
    credit_score, country, gender, age, tenure, balance,
    products_number, credit_card, active_member, estimated_salary,
    feature_cols
):
    # Base dict (raw features before dummies)
    row = {
        "credit_score": credit_score,
        "gender": 1 if gender == "Male" else 0,            # Male=1, Female=0
        "age": age,
        "tenure": tenure,
        "balance": balance,
        "products_number": products_number,
        "credit_card": 1 if credit_card == "Yes" else 0,
        "active_member": 1 if active_member == "Yes" else 0,
        "estimated_salary": estimated_salary,
        "country": country
    }

    df = pd.DataFrame([row])

    # One-hot encode country with drop_first=True (France as base in training)
    df = pd.get_dummies(df, columns=["country"], drop_first=True)
    # Ensure expected dummy columns exist; add missing with 0
    for col in ["country_Germany", "country_Spain"]:
        if col not in df.columns:
            df[col] = 0

    # Arrange columns to match training (feature_cols)
    df = df.reindex(columns=feature_cols, fill_value=0)

    return df

# ---------- Predict ----------
if submitted:
    X = preprocess_single_row(
        credit_score, country, gender, age, tenure, balance,
        products_number, credit_card, active_member, estimated_salary,
        feature_cols
    )

    # Scale
    X_scaled = scaler.transform(X)

    # Predict proba & class
    prob = float(model.predict_proba(X_scaled)[:, 1][0])
    pred = int(prob >= threshold)

    st.subheader("Results")
    st.metric(label="Churn Probability", value=f"{prob:.3f}")
    st.metric(label="Prediction", value=("Churn" if pred == 1 else "Stay"))

    # Simple guidance
    if prob >= 0.5:
        st.info("⚠️ Higher churn risk. Consider cross-sell/retention offers, reach-outs, or service recovery.")
    else:
        st.success("✅ Lower churn risk. Focus on engagement and product adoption.")

    # Show model inputs (debug)
    with st.expander("See processed feature vector (debug)"):
        st.write(pd.DataFrame(X, index=["input_row"]))

