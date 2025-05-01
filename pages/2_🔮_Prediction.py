import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the saved model and scaler
with open('model/random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title('🔮 Customer Churn Prediction')

# Create input fields
st.header('Customer Information')

# Demographics
col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox('Gender', ['Male', 'Female'])
    senior_citizen = st.selectbox('Senior Citizen', ['No', 'Yes'])
    partner = st.selectbox('Partner', ['No', 'Yes'])
    dependents = st.selectbox('Dependents', ['No', 'Yes'])

# Services
st.subheader('Services Information')
col3, col4 = st.columns(2)
with col3:
    phone_service = st.selectbox('Phone Service', ['No', 'Yes'])
    multiple_lines = st.selectbox('Multiple Lines', ['No', 'No phone service', 'Yes'])
    internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    online_security = st.selectbox('Online Security', ['No', 'Yes', 'No internet service'])
    online_backup = st.selectbox('Online Backup', ['No', 'Yes', 'No internet service'])

with col4:
    device_protection = st.selectbox('Device Protection', ['No', 'Yes', 'No internet service'])
    tech_support = st.selectbox('Tech Support', ['No', 'Yes', 'No internet service'])
    streaming_tv = st.selectbox('Streaming TV', ['No', 'Yes', 'No internet service'])
    streaming_movies = st.selectbox('Streaming Movies', ['No', 'Yes', 'No internet service'])

# Contract Information
st.subheader('Contract Information')
col5, col6 = st.columns(2)
with col5:
    contract = st.selectbox('Contract Type', ['Month-to-month', 'One year', 'Two year'])
    paperless_billing = st.selectbox('Paperless Billing', ['No', 'Yes'])
    payment_method = st.selectbox('Payment Method', 
                                ['Electronic check', 'Mailed check', 
                                 'Bank transfer (automatic)', 'Credit card (automatic)'])

with col6:
    tenure = st.number_input('Tenure (months)', min_value=0, max_value=100, value=12)
    monthly_charges = st.number_input('Monthly Charges ($)', min_value=0.0, max_value=1000.0, value=50.0)
    total_charges = st.number_input('Total Charges ($)', min_value=0.0, max_value=10000.0, value=600.0)

def preprocess_input(data):
    # Convert categorical variables
    data['gender'] = 1 if data['gender'] == 'Male' else 0
    data['SeniorCitizen'] = 1 if data['SeniorCitizen'] == 'Yes' else 0
    data['Partner'] = 1 if data['Partner'] == 'Yes' else 0
    data['Dependents'] = 1 if data['Dependents'] == 'Yes' else 0
    data['PhoneService'] = 1 if data['PhoneService'] == 'Yes' else 0
    data['PaperlessBilling'] = 1 if data['PaperlessBilling'] == 'Yes' else 0
    
    # Convert MultipleLines
    multiple_lines_map = {'Yes': 2, 'No': 1, 'No phone service': 0}
    data['MultipleLines'] = multiple_lines_map[data['MultipleLines']]
    
    # Convert InternetService
    internet_service_map = {'DSL': 2, 'No': 0, 'Fiber optic': 1}
    data['InternetService'] = internet_service_map[data['InternetService']]
    
    # Convert online services
    online_service_map = {'Yes': 2, 'No': 1, 'No internet service': 0}
    for col in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                'TechSupport', 'StreamingTV', 'StreamingMovies']:
        data[col] = online_service_map[data[col]]
    
    # Convert Contract
    contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
    data['Contract'] = contract_map[data['Contract']]
    
    # Convert PaymentMethod
    payment_map = {
        'Electronic check': 0, 
        'Mailed check': 1,
        'Bank transfer (automatic)': 2,
        'Credit card (automatic)': 3
    }
    data['PaymentMethod'] = payment_map[data['PaymentMethod']]
    
    return data

if st.button('Predict Churn'):
    # Create input data dictionary
    input_data = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    
    # Preprocess the input
    processed_data = preprocess_input(input_data)
    
    # Convert to DataFrame
    input_df = pd.DataFrame([processed_data])
    
    # Scale only the numeric features
    numeric_features = ['TotalCharges', 'MonthlyCharges', 'tenure']
    input_df[numeric_features] = scaler.transform(input_df[numeric_features])
    
    # Make prediction
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)
    
    # Display results
    st.header('Prediction Results')
    if prediction[0] == 1:
        st.error('⚠️ This customer is likely to churn!')
        st.write(f'Probability of churning: {probability[0][1]:.2%}')
    else:
        st.success('✅ This customer is likely to stay!')
        st.write(f'Probability of staying: {probability[0][0]:.2%}')