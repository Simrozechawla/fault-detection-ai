import streamlit as st
import joblib
import pandas as pd
import os

# ============================
# PATH SETUP (FIXED)
# ============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "model.pkl")
columns_path = os.path.join(BASE_DIR, "columns.pkl")

model = joblib.load(model_path)
feature_order = joblib.load(columns_path)

# ============================
# PAGE CONFIG
# ============================

st.set_page_config(page_title="AI Fault Detection", layout="centered")

st.title("🔧 AI-Based Fault Detection System")
st.markdown("Predict machine failures using sensor data")

# ============================
# SIDEBAR
# ============================

st.sidebar.title("About")
st.sidebar.info("""
AI-based predictive maintenance system.

Model: Random Forest  
Handles class imbalance (SMOTE)  
Deployed using Streamlit  
""")

# ============================
# INPUT SECTION
# ============================

st.subheader("Enter Machine Parameters")

col1, col2 = st.columns(2)

with col1:
    air_temp = st.number_input("Air Temperature (K)", value=300.0)
    process_temp = st.number_input("Process Temperature (K)", value=310.0)
    rpm = st.number_input("Rotational Speed (rpm)", value=1500.0)

with col2:
    torque = st.number_input("Torque (Nm)", value=40.0)
    tool_wear = st.number_input("Tool Wear (min)", value=0.0)
    type_input = st.selectbox("Machine Type", ["L", "M", "H"])

# ============================
# INPUT PREPARATION
# ============================

input_dict = {
    'Air temperature [K]': air_temp,
    'Process temperature [K]': process_temp,
    'Rotational speed [rpm]': rpm,
    'Torque [Nm]': torque,
    'Tool wear [min]': tool_wear,
    'Type_L': 1 if type_input == "L" else 0,
    'Type_M': 1 if type_input == "M" else 0,
    'Type_H': 1 if type_input == "H" else 0
}

input_data = pd.DataFrame([input_dict])

# Align EXACTLY with training columns
input_data = input_data.reindex(columns=feature_order, fill_value=0)

# ============================
# PREDICTION
# ============================

if st.button("Predict", key="predict_button"):
    try:
        prediction = model.predict(input_data)[0]

        result = prediction

        probs = model.predict_proba(input_data)
        confidence = max(probs[0]) * 100

        st.markdown("---")

        if result != "No Failure":
            st.error(f"⚠️ {result} Detected!")
        else:
            st.success("✅ Machine Operating Normally")

        st.info(f"Confidence: {confidence:.2f}%")

    except Exception as e:
        st.error(f"Error during prediction: {e}")

# ============================
# FOOTER
# ============================

st.markdown("---")
st.caption("Built by Simroze Chawla")