import streamlit as st
import tensorflow as tf
import numpy as np
import joblib

model = joblib.load('diabetes_model.h5') 

st.set_page_config(page_title="Diabetes Classification", layout="centered")
st.title("Diabetes Classification App")
st.write("Please enter your medical data.")

pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0)
glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=100)
blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)
skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=1000, value=100)
bmi = st.number_input("Body Mass Index (BMI)", min_value=0.0, max_value=50.0, value=25.0)
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
age = st.number_input("Age (years)", min_value=0, max_value=120, value=30)

if st.button("Check Result"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])

    prediction = model.predict(input_data)

    if prediction.ndim == 1: 
        predicted_class = int(prediction[0] > 0.5)
    else: 
        predicted_class = int(prediction[0][0] > 0.5)

    if predicted_class == 1:
        st.error("🚨 You are likely to have diabetes. Please consult a healthcare professional.")
    else:
        st.success("✅ You are likely not to have diabetes. Keep maintaining a healthy lifestyle!")

