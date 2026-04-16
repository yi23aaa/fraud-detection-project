# Fraud Detection System

A machine learning system for detecting fraudulent financial transactions using data analytics techniques

## Project Overview
This project develops an advanced fraud detection system by analysing historical transaction data and building models capable of classifying transactions as fraudulent or legitimate in real time

## Dataset
- Source: Kaggle Credit Card Fraud Detection Dataset
- 284,807 transactions with 492 fraud cases (0.17%)
- Features: 28 PCA components + Amount + Time

## Models Implemented
- Logistic Regression (baseline)
- Random Forest
- XGBoost (best performer)

## Key Results
| Model | ROC-AUC | Precision | Recall | F1-Score |
| Logistic Regression | 0.9698 | 0.0581 | 0.9184 | 0.1094 |
| Random Forest | 0.9731 | 0.8454 | 0.8367 | 0.8410 |
| XGBoost | 0.9792 | 0.7311 | 0.8878 | 0.8018 |

**Best Model: XGBoost with optimised threshold of 0.86**
- Final Precision: 86%
- Final Recall: 85%
- Final F1-Score: 0.8513

## Techniques Used
- SMOTE for class imbalance handling
- Decision threshold tuning
- Feature importance analysis
- Real-time transaction prediction simulation

## How to Run
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Add `creditcard.csv` to the project folder (not included due to size)
4. Open `fraud_detection.ipynb` and run all cells
cd C:\Users\yusuf\fraud-detection-project
conda activate fraud-env
streamlit run app.py

## Requirements
- Python 3.11
- pandas, numpy, matplotlib, seaborn
- scikit-learn, imbalanced-learn, xgboost
- jupyter, joblib
