import streamlit as st
import pickle
import pandas as pd

# App Config
st.set_page_config(page_title="Depression Prediction")
st.title("🧠 Depression Risk Prediction")

# Load Model
filename = 'Covid_model (1).pkl'
with open(filename, 'rb') as file:
    Model = pickle.load(file)

st.header("Input Features")

# ---- Input Fields ----
age = st.number_input("Age", min_value=10, max_value=100, step=1)

academic_pressure = st.slider("Academic Pressure (1-10)", 1, 10, 5)
work_pressure = st.slider("Work Pressure (1-10)", 1, 10, 5)
cgpa = st.number_input("CGPA (0-10)", min_value=0.0, max_value=10.0, step=0.1)
study_satisfaction = st.slider("Study Satisfaction (1-10)", 1, 10, 5)
work_study_hours = st.number_input("Work/Study Hours per day", min_value=0.0, max_value=24.0, step=0.5)

gender = st.selectbox("Gender", ["Female", "Male"])
gender_mapping = {'Female': 0, 'Male': 1}
gender_encoded = gender_mapping[gender]

sleep_duration = st.selectbox("Sleep Duration", ["<5 hrs", "5-7 hrs", ">7 hrs"])
sleep_mapping = {"<5 hrs": 0, "5-7 hrs": 1, ">7 hrs": 2}
sleep_duration_encoded = sleep_mapping[sleep_duration]

suicidal_thoughts = st.selectbox("Suicidal Thoughts", ["No", "Yes"])
suicidal_mapping = {'No': 0, 'Yes': 1}
suicidal_encoded = suicidal_mapping[suicidal_thoughts]

# ---- Prepare DataFrame ----
input_data = {
    "Age": [age],
    "Academic Pressure": [academic_pressure],
    "Work Pressure": [work_pressure],
    "CGPA": [cgpa],
    "Study Satisfaction": [study_satisfaction],
    "Work/Study Hours": [work_study_hours],
    "gender_encoded": [gender_encoded],
    "Sleep Duration_encoded": [sleep_duration_encoded],
    "suicidal thoughts": [suicidal_encoded]
}

test_df = pd.DataFrame(input_data)

# ---- Prediction ----
if st.button("Predict"):
    prediction = Model.predict(test_df)
    prediction_proba = Model.predict_proba(test_df)

    st.subheader("Prediction Result")
    if int(prediction[0]) == 0:
        st.success("🙂 Low Risk of Depression")
    else:
        st.warning("😟 High Risk of Depression")

    st.write("Prediction Probabilities:")
    st.write(prediction_proba)
