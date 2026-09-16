# 🎓 Digital Learning Analytics: Predicting Student Course Completion

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F18E19?style=flat&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1.5%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end Machine Learning project designed to predict whether a student on a digital learning platform will complete a course or drop out. By identifying at-risk students early based on engagement and academic metrics, platforms can provide timely interventions to improve retention rates.
📌 Project Overview & Problem StatementHigh dropout rates (~69% in our dataset) present a critical challenge for online learning platforms. The objective of this project is to build an automated classification pipeline that evaluates student behavioral and academic features to classify learners into:True (Completed): Students expected to complete the course (~31%).False (Not Completed / Dropped Out): Students at risk of dropping out (~69%).🚀 Key Results & Best ModelWe evaluated 8 Machine Learning classification algorithms. XGBoost achieved the best performance across all metrics:Best Model: XGBoost ClassifierROC-AUC Score: 92.56% 🏆Accuracy: 88.37%F1-Score: 0.7926Primary Predictive Feature: engagement_consistency (Contributes ~72% to the model's decisions).🛠️ Tech Stack & LibrariesLanguage: Python 3.8+Data Manipulation: pandas, numpyData Visualization: matplotlib, seabornMachine Learning: scikit-learn, xgboost🔄 Machine Learning PipelinePlaintext  1. Data Loading & Inspection
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
Exploratory Data Analysis (EDA):Handled 16,092 missing values using median imputation (robust to outliers).Addressed class imbalance (69% dropout vs. 31% completion) using class_weight='balanced'.Feature Engineering & Leakage Prevention:Engineered days_active from enrollment and last activity dates.Dropped post-course features (skill_post_score, mastery_score, time_to_mastery_hours) to prevent Data Leakage.Dropped non-predictive identifiers (learner_id, country).Data Preprocessing:Categorical columns encoded with LabelEncoder.Scaled numeric features using StandardScaler fitted strictly on train data (80/20 train-test split).📊 Model ComparisonRankModelAccuracyF1-ScoreROC-AUC1️⃣XGBoost0.88370.79260.92562️⃣SVM0.87160.79010.92083️⃣Logistic Regression0.85480.77650.91744️⃣Random Forest0.86960.78280.91455️⃣Bagging Classifier0.87910.78260.91296️⃣Naive Bayes0.86210.75920.90037️⃣Decision Tree0.86940.76770.89258️⃣K-Nearest Neighbors (KNN)0.83310.64540.8695👥 Team Presentation & RolesMemberHeadlineCovered TopicsMember 1Introduction & Dropout ProblemProject goals, Problem statement & Pipeline overviewMember 2Data Preprocessing & EDALibraries, Median Imputation & Handling Class ImbalanceMember 3Feature Engineering & Data PrepCorrelation Heatmap, Preventing Data Leakage & ScalingMember 4Model Training & XGBoost TuningComparing 8 Models & XGBoost HyperparametersMember 5Results Evaluation & ConclusionROC-AUC results, Confusion Matrix, Feature Importance & Recommendations📈 Key Insights & RecommendationsConsistency > Volume: engagement_consistency accounts for ~72% of the predictive importance. Consistent daily/weekly engagement is a far stronger predictor of success than occasional high-volume learning sessions.Early Intervention: Integrating the XGBoost model into an automated early-warning system enables platform administrators to target disengaged students with personalized nudges before they abandon the course.📂 Repository StructurePlaintext.
├── presentation/
│   └── Digital_Learning_Analytics_ML_Presentation.pptx   # Presentation Deck
├── notebooks/
│   └── course_completion_ml.ipynb                        # Full ML Pipeline & Analysis
├── README.md                                             # Project Documentation
└── requirements.txt                                      # Python Dependencies
💻 How to RunClone the repository:Bashgit clone [https://github.com/your-username/digital-learning-analytics.git](https://github.com/your-username/digital-learning-analytics.git)
cd digital-learning-analytics
Install dependencies:Bashpip install -r requirements.txt
Run Jupyter Notebook:Bashjupyter notebook notebooks/course_completion_ml.ipynb
