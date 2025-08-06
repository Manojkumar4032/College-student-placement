import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("placement_model.sav")
scaler = joblib.load("scaler.sav")

st.title("🎓 Student Placement Prediction App")

st.markdown("Fill the form below to predict if a student will be placed or not.")

# Input form
iq = st.number_input("IQ", min_value=50, max_value=200, value=100)
prev_sem_result = st.number_input("Previous Semester Result", min_value=0.0, max_value=10.0, value=6.0)
cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=6.5)
academic_perf = st.slider("Academic Performance (1-10)", 1, 10, 7)
internship = st.selectbox("Internship Experience", ["Yes", "No"])
extra_curricular = st.slider("Extra Curricular Score (0-10)", 0, 10, 5)
comm_skills = st.slider("Communication Skills (0-10)", 0, 10, 5)
projects = st.number_input("Projects Completed", min_value=0, max_value=20, value=2)

# Preprocess input
internship_val = 1 if internship == "Yes" else 0

input_data = np.array([[iq, prev_sem_result, cgpa, academic_perf,
                        internship_val, extra_curricular, comm_skills, projects]])

input_scaled = scaler.transform(input_data)

# Predict button
if st.button("Predict Placement"):
    prediction = model.predict(input_scaled)
    result = "🎉 Placed" if prediction[0] == 1 else "❌ Not Placed"
    st.subheader(f"Prediction: {result}")