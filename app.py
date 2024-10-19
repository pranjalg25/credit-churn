pip3 install tensorflow
import streamlit as st
import numpy as np
from keras.models import load_model
import joblib

# Load the trained model and scaler
model = load_model('saved_model.h5')
scaler = joblib.load('scaler.pkl')

# Streamlit app interface
st.title("Customer Churn Prediction")
st.write("Enter the customer's details for churn prediction:")

# Input features
CreditScore = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
Age = st.number_input("Age", min_value=18, max_value=100, value=30)
Tenure = st.number_input("Tenure (years)", min_value=0, max_value=10, value=5)
Balance = st.number_input("Account Balance", min_value=0.0, max_value=250000.0, value=50000.0)
NumOfProducts = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
HasCrCard = st.selectbox("Has Credit Card?", [0, 1])  # 0 = No, 1 = Yes
IsActiveMember = st.selectbox("Is Active Member?", [0, 1])  # 0 = No, 1 = Yes
EstimatedSalary = st.number_input("Estimated Salary", min_value=0.0, max_value=200000.0, value=50000.0)

# Geography input
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
Geography_Germany = 1 if geography == "Germany" else 0
Geography_Spain = 1 if geography == "Spain" else 0

# Gender input
gender = st.selectbox("Gender", ["Male", "Female"])
Gender_Male = 1 if gender == "Male" else 0

# When the 'Predict' button is clicked
if st.button("Predict"):
    # Prepare input features for prediction
    features = np.array([[CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, 
                          IsActiveMember, EstimatedSalary, Geography_Germany, 
                          Geography_Spain, Gender_Male]])

    # Scale the input features
    features = scaler.transform(features)

    # Make prediction
    prediction = model.predict(features)
    predict_class = np.where(prediction > 0.5, 1, 0)

    # Output the prediction result
    if predict_class[0][0] == 1:
        st.write("Prediction: The customer is likely to churn.")
    else:
        st.write("Prediction: The customer is not likely to churn.")

sample_features = np.array([[675, 36, 9, 106190.55, 1, 0, 1,  22994.32, 0, 1, 0]])  # Adjust these values as necessary
prediction = model.predict(sample_features)
print(prediction)
if prediction[0] == 1:
    print("The customer is likely to churn.")
else:
    print("The customer is not likely to churn.")
