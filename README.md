# 🎓 Digital Learning Analytics: Predicting Student Course Completion

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F18E19?style=flat&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1.5%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)
📌 Project Overview & Problem Statement
High dropout rates (~69% in our dataset) present a critical challenge for online learning platforms. The objective of this project is to build an automated classification pipeline that evaluates student behavioral and academic features to classify learners into:

True (Completed): Students expected to complete the course (~31%).

False (Not Completed / Dropped Out): Students at risk of dropping out (~69%).

🚀 Key Results & Best Model
We evaluated 8 Machine Learning classification algorithms. XGBoost achieved the best performance across all metrics:

Best Model: XGBoost Classifier

ROC-AUC Score: 92.56% 🏆

Accuracy: 88.37%

F1-Score: 0.7926

Primary Predictive Feature: engagement_consistency (Contributes ~72% to the model's decisions).

🛠️ Tech Stack & Libraries
Language: Python 3.8+

Data Manipulation: pandas, numpy

Data Visualization: matplotlib, seaborn

Machine Learning: scikit-learn, xgboost


🔄 Machine Learning Pipeline
1. Data Loading & Inspection
             │
             ▼
  2. EDA & Imputation (Median for Outliers)
             │
             ▼
  3. Data Leakage Prevention & Feature Selection
             │
             ▼
  4. Encoding & Feature Scaling (StandardScaler)
             │
             ▼
  5. Training 8 Classification Models
             │
             ▼
  6. Model Evaluation (ROC-AUC, Confusion Matrix)








