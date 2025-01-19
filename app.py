import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit App Title
st.title('Customer Churn Prediction')

# User Input Section
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])  # Dropdown for Geography
gender = st.selectbox('Gender', label_encoder_gender.classes_)            # Dropdown for Gender
age = st.slider('Age', 18, 92)                                            # Slider for Age
balance = st.number_input('Balance', min_value=0.0, step=1000.0)          # Number input for Balance
credit_score = st.number_input('Credit Score', min_value=300.0, step=1.0) # Number input for Credit Score
estimated_salary = st.number_input('Estimated Salary', min_value=0.0, step=1000.0)  # Number input for Salary
tenure = st.slider('Tenure', 0, 10)                                       # Slider for Tenure
num_of_products = st.slider('Number of Products', 1, 4)                   # Slider for Number of Products
has_cr_card = st.selectbox('Has Credit Card', [0, 1])                     # Dropdown for Credit Card Ownership
is_active_member = st.selectbox('Is Active Member', [0, 1])               # Dropdown for Active Membership

# Prepare Input Data for Prediction
# Encode Gender
gender_encoded = label_encoder_gender.transform([gender])[0]

# One-hot encode Geography
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(
    geo_encoded, 
    columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
)

# Combine all inputs into a DataFrame
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender_encoded],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# Concatenate the one-hot encoded Geography data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Make Prediction
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]  # Extract the probability from the prediction

# Display Results
st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')
