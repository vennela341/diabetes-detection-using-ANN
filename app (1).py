import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

model = load_model("model.keras")
scaler = joblib.load("scaler.pk1")

st.title("🩺 Diabetes Prediction using ANN")
st.write("Enter Patient Details")

# Diabetes dataset inputs
pregnancies = st.number_input("Pregnancies", 0, 20)
glucose = st.number_input("Glucose Level", 0, 300)
blood_pressure = st.number_input("Blood Pressure", 0, 200)
skin_thickness = st.number_input("Skin Thickness", 0, 100)
insulin = st.number_input("Insulin Level", 0, 900)
bmi = st.number_input("BMI", 0.0, 70.0)
dpf = st.number_input("Diabetes Pedigree Function (DPF)", 0.0, 3.0)
age = st.number_input("Age", 1, 120)

if st.button("Predict"):

    input_data = np.array([[pregnancies, glucose, blood_pressure,
                            skin_thickness, insulin, bmi, dpf, age]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    if prediction[0][0] > 0.5:
        st.error("⚠️ Diabetes Detected")
    else:
        st.success("✅ No Diabetes Detected")