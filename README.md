📉 Telco Customer Churn — Classification with Random Forest
Mostrar imagen
Mostrar imagen
Mostrar imagen
End-to-end machine learning project that predicts customer churn for a telecommunications company using a Random Forest Classifier. Includes full EDA, statistical hypothesis testing, data preprocessing, PCA, model training with hyperparameter tuning, and business-oriented insights.

📊 Project Overview
DatasetIBM Telco Customer Churn — 7,043 customers, 21 featuresTarget variableChurn (0 = No, 1 = Yes)ModelRandom Forest Classifier (GridSearchCV optimized)ROC-AUC (test)0.843F1-Score (test)0.628Accuracy (test)0.770

🔬 Methodology
1. Exploratory Data Analysis

Descriptive statistics and class balance diagnosis (26.5% churn rate)
Distribution analysis of numerical variables: tenure, MonthlyCharges, TotalCharges
Churn rate breakdown by categorical variables: Contract, InternetService, PaymentMethod, TechSupport
Correlation heatmap (tenure r = −0.352, MonthlyCharges r = +0.193 with Churn)

2. Data Cleaning & Encoding

TotalCharges converted from string to numeric; 11 nulls imputed with median
Binary variables (Yes/No) mapped to 0/1
One-Hot Encoding for InternetService, Contract, and PaymentMethod (drop_first=True)

3. Hypothesis Testing (before model training)

Welch t-test (MonthlyCharges): churners pay $74.44 vs $61.27 → t = 18.41, p ≈ 8.59e-73 ✓
Welch t-test (tenure): churners stay 18.0 vs 37.6 months → t = −34.82, p ≈ 1.20e-232 ✓
One-way ANOVA (Contract type): F = 20.83, p ≈ 9.58e-10 ✓

4. Preprocessing

StandardScaler applied to continuous features (tenure, MonthlyCharges, TotalCharges)
Scaler fit only on train set to prevent data leakage
Train/test split: 80/20 stratified by Churn (5,634 / 1,409 observations)

5. PCA — Dimensionality Reduction

Full PCA fit on training data to analyze explained variance
2 components capture only 56.2% of total variance → information is distributed across many dimensions
2D projection confirms classes are not linearly separable → justifies non-linear model

6. Model Training

Random Forest with GridSearchCV (5-fold StratifiedKFold, scoring = F1)
Best config: class_weight='balanced', max_depth=10, min_samples_split=2, n_estimators=100
Best F1 (CV): 0.634 — consistent with test F1 of 0.628 (no overfitting)


📈 Key Results
MetricValueAccuracy (test)0.770F1-Score (test)0.628ROC-AUC (test)0.843Best F1 (CV)0.634
Top predictors by feature importance
VariableImportanceBusiness insighttenure0.178Newer customers churn most — first 12 months are criticalTotalCharges0.141Low accumulated spend = low switching costMonthlyCharges0.125High monthly cost drives churn when value isn't perceivedContract_Two year0.107Long-term contracts are the strongest retention shieldInternetService_Fiber optic0.076Fiber customers have highest expectations and churn rate
Key business insights

Tenure is the #1 predictor (importance = 0.178, r = −0.352): churn risk peaks in the first year. Retention programs should prioritize onboarding and early engagement.
Month-to-month contracts churn at 42.7% vs 2.8% for two-year contracts (ANOVA p ≈ 0.00). Incentivizing long-term contracts is the single most actionable retention lever.
Churners pay $13.18 more per month on average (Welch t-test p ≈ 0.00) — a signal of price-value disconnect in high-spend segments.
TechSupport and OnlineSecurity reduce churn — customers embedded in the service ecosystem are harder to lose.


🛠 Tech Stack
pandas · numpy · matplotlib · seaborn · scikit-learn · scipy

🚀 Getting Started
Clone the repo and install dependencies:
bashpip install pandas numpy matplotlib seaborn scikit-learn scipy
Download the dataset from Kaggle and place it in the root folder as WA_Fn-UseC_-Telco-Customer-Churn.csv:
🔗 https://www.kaggle.com/datasets/blastchar/telco-customer-churn
Then open and run the notebook:
telco_churn_classification.ipynb

📁 Notebook Structure
SectionDescriptionPart 1 — EDADataset overview, class balance, distributions, churn rate by category, correlation heatmapPart 2 — CleaningType correction, null imputation, binary encoding, One-Hot EncodingPart 3 — Hypothesis TestingWelch t-test (MonthlyCharges, tenure), one-way ANOVA (Contract)Part 4 — PCAFull variance analysis, scree plot, 2D class separation visualizationPart 5 — Random ForestGridSearchCV tuning, evaluation metrics, confusion matrix, ROC curve, feature importancePart 6 — ConclusionsExecutive summary with business-oriented recommendations

🔮 Future Improvements

Gradient Boosting (XGBoost / LightGBM): expected to push ROC-AUC above 0.90 on this dataset
Classification threshold tuning: adjusting the decision threshold (default 0.5) based on the business cost of false negatives vs false positives
SMOTE: oversampling the minority class to improve recall without sacrificing ROC-AUC
Temporal features: monthly usage history and support contact frequency would significantly increase predictive power in a production environment


📚 References

IBM. (2018). Telco Customer Churn [Dataset]. Kaggle. https://www.kaggle.com/datasets/blastchar/telco-customer-churn


👤 Author
Gerardo Olivares
Software Engineering student @ Tecmilenio | ML Diploma
GitHub
Built as part of a Machine Learning Diploma portfolio project.
