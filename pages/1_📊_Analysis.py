import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")

# Load and cache the data
@st.cache_data
def load_data():
    df = pd.read_csv("datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    return df

st.title('📊 Customer Data Analysis')

# Load the data
df = load_data()

# Sidebar for filtering
st.sidebar.header('Data Filters')
contract_filter = st.sidebar.multiselect(
    'Contract Type',
    options=df['Contract'].unique(),
    default=df['Contract'].unique()
)

internet_filter = st.sidebar.multiselect(
    'Internet Service',
    options=df['InternetService'].unique(),
    default=df['InternetService'].unique()
)

# Filter the data
filtered_df = df[
    (df['Contract'].isin(contract_filter)) &
    (df['InternetService'].isin(internet_filter))
]

# Overview metrics
st.header('Overview Metrics')
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_customers = len(filtered_df)
    st.metric("Total Customers", f"{total_customers:,}")

with col2:
    churn_rate = (filtered_df['Churn'] == 'Yes').mean() * 100
    st.metric("Churn Rate", f"{churn_rate:.1f}%")

with col3:
    avg_tenure = filtered_df['tenure'].mean()
    st.metric("Avg. Tenure (months)", f"{avg_tenure:.1f}")

with col4:
    avg_monthly = filtered_df['MonthlyCharges'].mean()
    st.metric("Avg. Monthly Charges", f"${avg_monthly:.2f}")

# Visualizations
st.header('Data Visualizations')

# Row 1: Churn Analysis
row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    st.subheader('Churn by Contract Type')
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    contract_churn = filtered_df.groupby(['Contract', 'Churn']).size().unstack()
    contract_churn.plot(kind='bar', stacked=True, ax=ax1)
    plt.title('Customer Churn by Contract Type')
    plt.xlabel('Contract Type')
    plt.ylabel('Number of Customers')
    plt.legend(title='Churn')
    st.pyplot(fig1)

with row1_col2:
    st.subheader('Churn by Internet Service')
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    internet_churn = filtered_df.groupby(['InternetService', 'Churn']).size().unstack()
    internet_churn.plot(kind='bar', stacked=True, ax=ax2)
    plt.title('Customer Churn by Internet Service')
    plt.xlabel('Internet Service')
    plt.ylabel('Number of Customers')
    plt.legend(title='Churn')
    st.pyplot(fig2)

# Row 2: Payment and Services Analysis
row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    st.subheader('Monthly Charges Distribution')
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=filtered_df, x='Contract', y='MonthlyCharges', hue='Churn', ax=ax3)
    plt.title('Monthly Charges by Contract Type and Churn Status')
    plt.xticks(rotation=45)
    st.pyplot(fig3)

with row2_col2:
    st.subheader('Services Distribution')
    services = ['PhoneService', 'InternetService', 'OnlineSecurity', 
               'OnlineBackup', 'DeviceProtection', 'TechSupport']
    service_data = filtered_df[services].melt()
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.countplot(data=service_data, x='variable', hue='value', ax=ax4)
    plt.title('Services Distribution')
    plt.xticks(rotation=45)
    st.pyplot(fig4)

# Row 3: Tenure Analysis
st.subheader('Tenure Analysis')
fig5, ax5 = plt.subplots(figsize=(12, 6))
sns.histplot(data=filtered_df, x='tenure', hue='Churn', multiple="stack", bins=30, ax=ax5)
plt.title('Customer Tenure Distribution by Churn Status')
st.pyplot(fig5)

# Detailed Analysis
st.header('Detailed Analysis')
if st.checkbox('Show Raw Data'):
    st.write(filtered_df)

# Feature Correlations
st.subheader('Feature Correlations')
numeric_df = filtered_df.select_dtypes(include=[np.number])
fig6, ax6 = plt.subplots(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax6)
plt.title('Correlation Matrix of Numeric Features')
st.pyplot(fig6)