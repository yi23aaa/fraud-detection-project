import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import time

st.set_page_config(page_title="Fraud Detection System", layout="wide")

# Load model
@st.cache_resource
def load_models():
    model = joblib.load('best_model_xgboost.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_cols = joblib.load('feature_columns.pkl')
    return model, scaler, feature_cols

@st.cache_data
def load_data():
    return pd.read_csv('creditcard.csv')

model, scaler, feature_cols = load_models()
df = load_data()

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Predict Transaction"])

# ============================================================
# HOME PAGE
# ============================================================
if page == "Home":
    st.title("Fraud Detection System")
    st.write("A machine learning system for detecting fraudulent financial transactions.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Transactions", f"{len(df):,}")
    with col2:
        st.metric("Fraud Cases", f"{df['Class'].sum():,}")
    with col3:
        st.metric("Fraud Rate", f"{df['Class'].mean()*100:.2f}%")

    st.markdown("---")
    st.subheader("Class Distribution")

    fig, ax = plt.subplots(figsize=(6, 4))
    class_counts = df['Class'].value_counts()
    ax.bar(['Legitimate', 'Fraud'], class_counts.values,
           color=['steelblue', 'crimson'], edgecolor='black')
    ax.set_ylabel('Number of Transactions')
    ax.set_title('Transaction Class Distribution')
    for i, v in enumerate(class_counts.values):
        ax.text(i, v + 500, f'{v:,}', ha='center', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ============================================================
# PREDICT TRANSACTION PAGE
# ============================================================
elif page == "Predict Transaction":
    st.title("Predict Transaction")
    st.write("Enter transaction details below to classify as fraudulent or legitimate.")
    st.markdown("---")

    amount = st.number_input("Transaction Amount (£)", 
                              min_value=0.0, max_value=50000.0, 
                              value=100.0, step=0.01)
    time_val = st.number_input("Time (seconds since first transaction)", 
                                min_value=0.0, max_value=200000.0, 
                                value=50000.0, step=1.0)

    st.markdown("**PCA Features (V1-V28):**")
    v_values = {}
    cols = st.columns(7)
    for i in range(1, 29):
        with cols[(i-1) % 7]:
            v_values[f'V{i}'] = st.number_input(f'V{i}', value=0.0,
                                                  format="%.4f",
                                                  key=f'v{i}')

    st.markdown("---")
    if st.button("Predict", type="primary"):
        transaction = {'Time': time_val, 'Amount': amount}
        transaction.update(v_values)

        transaction_df = pd.DataFrame([transaction])[feature_cols]
        transaction_df[['Amount', 'Time']] = scaler.transform(
            transaction_df[['Amount', 'Time']]
        )

        fraud_prob = model.predict_proba(transaction_df)[0][1]
        is_fraud = fraud_prob >= 0.86

        with st.spinner("Analysing transaction..."):
            time.sleep(0.5)

        st.markdown("---")
        st.subheader("Result")

        col1, col2, col3 = st.columns(3)
        with col1:
            if is_fraud:
                st.error("FRAUDULENT TRANSACTION")
            else:
                st.success("LEGITIMATE TRANSACTION")
        with col2:
            st.metric("Fraud Probability", f"{fraud_prob*100:.2f}%")
        with col3:
            if fraud_prob >= 0.86:
                st.metric("Risk Level", "HIGH RISK")
            elif fraud_prob >= 0.50:
                st.metric("Risk Level", "MEDIUM RISK")
            else:
                st.metric("Risk Level", "LOW RISK")

        st.info(f"The optimised decision threshold is 0.86 (86%). "
                f"This transaction scored {fraud_prob*100:.2f}%, which is "
                f"{'above' if is_fraud else 'below'} the threshold, "
                f"therefore classified as {'FRAUDULENT' if is_fraud else 'LEGITIMATE'}.")