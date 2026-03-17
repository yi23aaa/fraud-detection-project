import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
import seaborn as sns

st.set_page_config(page_title="Fraud Detection System", layout="wide")

#Load model
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

#Prepare test data for performance page
X = df.drop('Class', axis=1)
y = df['Class']
X_scaled = X.copy()
X_scaled[['Amount', 'Time']] = scaler.transform(X_scaled[['Amount', 'Time']])
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
y_prob_test = model.predict_proba(X_test)[:, 1]
y_pred_test = (y_prob_test >= 0.86).astype(int)

#Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Predict Transaction", "Model Performance", "Dataset Analysis"])

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
# ============================================================
# MODEL PERFORMANCE PAGE
# ============================================================
elif page == "Model Performance":
    st.title("Model Performance")
    st.write("Evaluation results for all models trained during this project.")
    st.markdown("---")

    st.subheader("Model Comparison")
    summary = pd.DataFrame({
        'Model': ['Logistic Regression', 'Decision Tree',
                  'Random Forest', 'XGBoost'],
        'ROC-AUC': [0.9698, 0.8951, 0.9731, 0.9792],
        'Precision (Fraud)': [0.0581, 0.0800, 0.8454, 0.7311],
        'Recall (Fraud)': [0.9184, 0.8100, 0.8367, 0.8878],
        'F1-Score (Fraud)': [0.1094, 0.1500, 0.8410, 0.8018]
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_prob_test)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, color='darkorange', lw=2,
                label='XGBoost (AUC = 0.9792)')
        ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve - XGBoost')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred_test)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Legitimate', 'Fraud'],
                    yticklabels=['Legitimate', 'Fraud'])
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        ax.set_title('Confusion Matrix - XGBoost (threshold=0.86)')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader("Cross Validation Results (5-Fold)")
    cv_data = pd.DataFrame({
        'Model': ['Logistic Regression', 'Decision Tree',
                  'Random Forest', 'XGBoost'],
        'Mean AUC': [0.9734, 0.8723, 0.9466, 0.9446],
        'Std Dev (+/-)': [0.0141, 0.0306, 0.0220, 0.0386]
    })
    st.dataframe(cv_data, use_container_width=True, hide_index=True)
# ============================================================
# DATASET ANALYSIS PAGE
# ============================================================
elif page == "Dataset Analysis":
    st.title("Dataset Analysis")
    st.write("Exploratory analysis of the credit card transaction dataset.")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Dataset Overview")
        st.write(f"**Total transactions:** {len(df):,}")
        st.write(f"**Legitimate:** {(df['Class']==0).sum():,} ({(df['Class']==0).mean()*100:.2f}%)")
        st.write(f"**Fraudulent:** {(df['Class']==1).sum():,} ({(df['Class']==1).mean()*100:.2f}%)")
        st.write(f"**Number of features:** {df.shape[1]-1}")
        st.write(f"**Missing values:** {df.isnull().sum().sum()}")

        st.markdown("---")
        st.subheader("Amount Statistics by Class")
        amount_stats = df.groupby('Class')['Amount'].describe().round(2)
        amount_stats.index = ['Legitimate', 'Fraud']
        st.dataframe(amount_stats, use_container_width=True)

    with col2:
        st.subheader("Transaction Amount Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        legit = df[df['Class'] == 0]['Amount']
        fraud = df[df['Class'] == 1]['Amount']
        ax.hist(legit[legit <= 500], bins=50, alpha=0.7,
                color='steelblue', label='Legitimate')
        ax.hist(fraud[fraud <= 500], bins=50, alpha=0.7,
                color='crimson', label='Fraud')
        ax.set_xlabel('Amount (£)')
        ax.set_ylabel('Frequency')
        ax.set_title('Amount Distribution (up to £500)')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Fraud Rate by Hour of Day")
        df_temp = df.copy()
        df_temp['Time_Hour'] = (df_temp['Time'] / 3600) % 24
        hourly_fraud = df_temp.groupby(
            df_temp['Time_Hour'].astype(int))['Class'].mean() * 100

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(hourly_fraud.index, hourly_fraud.values,
                color='crimson', lw=2, marker='o', markersize=4)
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Fraud Rate (%)')
        ax.set_title('Fraud Rate by Hour of Day')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Transaction Volume by Hour of Day")
        df_temp['Time_Hour'] = (df_temp['Time'] / 3600) % 24
        hourly_volume = df_temp.groupby(
            df_temp['Time_Hour'].astype(int)).size()

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(hourly_volume.index, hourly_volume.values,
               color='steelblue', edgecolor='black', alpha=0.8)
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Number of Transactions')
        ax.set_title('Transaction Volume by Hour of Day')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader("Batch Transaction Simulation")
    st.write("Simulate real-time monitoring on a stream of transactions.")

    n_transactions = st.slider("Number of transactions", 
                                min_value=10, max_value=50, value=20)
    fraud_count = st.slider("Fraud cases to include", 
                             min_value=1, max_value=10, value=5)

    if st.button("Run Simulation", type="primary"):
        fraud_indices = y_test[y_test == 1].sample(
            n=fraud_count, random_state=42).index
        legit_count = n_transactions - fraud_count
        legit_indices = y_test[y_test == 0].sample(
            n=legit_count, random_state=42).index
        sample_indices = fraud_indices.append(legit_indices)

        sample_X = X_test.loc[sample_indices].copy()
        sample_y = y_test.loc[sample_indices]

        shuffle_idx = sample_X.sample(frac=1, random_state=42).index
        sample_X = sample_X.loc[shuffle_idx]
        sample_y = sample_y.loc[shuffle_idx]

        probs = model.predict_proba(sample_X)[:, 1]
        preds = (probs >= 0.86).astype(int)

        results_data = []
        for actual, pred, prob in zip(sample_y, preds, probs):
            results_data.append({
                'Actual': 'FRAUD' if actual == 1 else 'LEGIT',
                'Predicted': 'FRAUD' if pred == 1 else 'LEGIT',
                'Fraud Probability (%)': round(prob * 100, 2),
                'Risk Level': 'HIGH' if prob >= 0.86 else (
                    'MEDIUM' if prob >= 0.50 else 'LOW'),
                'Status': 'CORRECT' if actual == pred else 'WRONG'
            })

        results_df = pd.DataFrame(results_data)

        def highlight_rows(row):
            if row['Actual'] == 'FRAUD' and row['Predicted'] == 'FRAUD':
                return ['background-color: #ffcccc'] * len(row)
            elif row['Status'] == 'WRONG':
                return ['background-color: #fff3cc'] * len(row)
            else:
                return [''] * len(row)

        st.dataframe(results_df.style.apply(highlight_rows, axis=1),
                     use_container_width=True)

        correct = (results_df['Status'] == 'CORRECT').sum()
        fraud_caught = ((results_df['Actual'] == 'FRAUD') &
                       (results_df['Predicted'] == 'FRAUD')).sum()
        false_positives = ((results_df['Actual'] == 'LEGIT') &
                          (results_df['Predicted'] == 'FRAUD')).sum()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy",
                      f"{correct}/{n_transactions} ({correct/n_transactions*100:.1f}%)")
        with col2:
            st.metric("Fraud Detected", f"{fraud_caught}/{fraud_count}")
        with col3:
            st.metric("False Positives",
                      f"{false_positives}/{n_transactions - fraud_count}")