# Bankruptcy Prediction in Taiwan’s Economy (1999–2009)

📉 Predicting corporate bankruptcy using Machine Learning techniques with real-world financial ratio data from the Taiwan Economic Journal.

## 📚 Project Summary

This project implements a machine learning-based framework to predict bankruptcy among Taiwanese firms from 1999–2009. We use:
- Feature selection (RFE, Logistic Stepwise, Economic Factors)
- Class imbalance handling (SMOTE and Down Sampling)
- Classifiers: Logistic Regression, Random Forest, Gradient Boosting, XGBoost

## 🔧 Key Features

- End-to-end pipeline for bankruptcy prediction
- Variable importance analysis
- SMOTE vs Downsampling performance comparison
- Model performance metrics: Accuracy, Precision, Recall, F1-Score, AUC

## 📊 Dataset

- Source: Taiwan Economic Journal (via Kaggle & UC Irvine)
- 10 years of financial ratio data (~96 variables)
- Imbalanced data: 6,599 non-bankrupt vs. few bankrupt firms

## 🧠 Models Used

- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost

## 🧪 Results Snapshot

| Model              | F1 Score | AUC (SMOTE) |
|-------------------|----------|-------------|
| Logistic Regression | 0.87     | 0.8590      |
| Random Forest       | 0.98     | 0.998       |
| Gradient Boosting   | 0.89     | 0.9059      |
| XGBoost             | **0.98** | **0.9728**  |

> Random Forest and XGBoost outperform other models, especially after SMOTE application.

## 📁 Repository Layout

- `data/`: Raw and processed datasets
- `notebooks/`: Jupyter notebooks for EDA, feature selection, modeling
- `src/`: Scripts for preprocessing, training, evaluation
- `models/`: Saved ML models
- `reports/`: Figures and summary reports

## 📦 Requirements

```bash
pip install -r requirements.txt
Includes:
•	pandas, numpy, scikit-learn, xgboost, seaborn, matplotlib, imbalanced-learn
