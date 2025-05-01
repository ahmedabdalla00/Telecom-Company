import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Telecom Customer Churn",
    page_icon="📊",
    layout="wide"
)

st.title('📊 Telecom Customer Churn Analysis')

st.write("""
Welcome to the Telecom Customer Churn Analysis Dashboard. This application helps you:
- Predict customer churn probability using our trained machine learning model
- Analyze customer data and identify patterns
- Visualize key metrics and trends in the dataset

Use the sidebar to navigate between different pages:
- 🔮 Prediction: Enter customer information to predict churn probability
- 📊 Analysis: Explore data visualizations and insights
""")

# Display some key metrics from the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    return df

df = load_data()

# Quick overview
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Customers", f"{len(df):,}")
    
with col2:
    churn_rate = (df['Churn'] == 'Yes').mean() * 100
    st.metric("Overall Churn Rate", f"{churn_rate:.1f}%")
    
with col3:
    avg_monthly = df['MonthlyCharges'].mean()
    st.metric("Avg. Monthly Revenue", f"${avg_monthly:.2f}")

st.sidebar.success("Select a page above.")