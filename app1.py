import streamlit as st
import pandas as pd
import joblib
# Load Saved Model, Scaler and Columns
model = joblib.load("salary_prediction_model.pkl")
scaler = joblib.load("scaler.pkl")
encoded_columns = joblib.load("columns.pkl")
# Streamlit Page Configuration
st.set_page_config(
    page_title="Salary Prediction",
    layout="centered"
)
# Title
st.title("Employee Salary Prediction")
st.write("Enter employee details below to predict salary.")
# Numeric Inputs
age = st.number_input(
    "Age",
    min_value=18,
    max_value=70,
    value=30
)

experience = st.number_input(
    "Years of Experience",
    min_value=0.0,
    max_value=50.0,
    value=5.0
)

# Categorical Inputs
gender = st.selectbox(
    "Gender",
    [
        "Male",
        "Female",
        "Other"
    ]
)

education = st.selectbox(
    "Education Level",
    [
        "High School",
        "Bachelor's",
        "Master's",
        "PhD"
    ]
)

job_title = st.text_input(
    "Job Title"
)

# Predict Button

predict = st.button("Predict Salary")

# Prediction
if predict:

    # Create DataFrame
    input_data = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Education Level": [education],
        "Job Title": [job_title],
        "Years of Experience": [experience]
    })

    # One-Hot Encoding
    input_encoded = pd.get_dummies(input_data)

    # Match Training Columns
    input_encoded = input_encoded.reindex(
        columns=encoded_columns,
        fill_value=0
    )

    # Scale Numeric Columns
    numeric_columns = [
        "Age",
        "Years of Experience"
    ]
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    input_encoded[numeric_columns] = scaler.fit_transform(
        input_encoded[numeric_columns]
    )

    # Prediction
    predicted_salary = model.predict(input_encoded)

    # Display Result
    st.success(
        f"Predicted Salary : ${predicted_salary[0]:,.2f}"
    )