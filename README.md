Diabetes Data Analysis

Overview
This project performs an in-depth analysis of diabetes data, involving data cleaning, exploratory data analysis, unsupervised learning, and machine learning model training. The goal is to identify patterns and build predictive models for diabetes diagnosis.

Features Implemented
Data Cleaning- Handling missing values by replacing zeros with median values.
Exploratory Data Analysis (EDA)- Correlation heatmap visualization, Dimensionality reduction using PCA, UMAP, and t-SNE.
Unsupervised Learning- K-Means clustering for pattern recognition.
Supervised Learning Models- Logistic Regression, Random Forest Classifier, XGBoost Classifier
Model Evaluation- Accuracy computation, Cross-validation for model performance assessment.
Hyperparameter Tuning- GridSearchCV for optimizing Random Forest parameters.
Feature Selection- Selecting the most important features using Random Forest.
Regularization Experiment- L2 Regularization applied to Logistic Regression.

Dataset
The dataset used is the diabetes.csv file, which contains relevant features for diabetes prediction.

Installation and Requirements

Dependencies:
Ensure you have the following Python libraries installed:
pip install pandas numpy seaborn matplotlib scikit-learn umap-learn xgboost

How to Run
Place diabetes.csv in the appropriate directory.
Run the Python script to execute the analysis:
python diabetes_analysis.py

Results
Logistic Regression Accuracy: 76.62%
Random Forest Accuracy: 76.62%
XGBoost Accuracy: 73.38%
Optimized Random Forest CV Accuracy: 78.02%
Feature Selection Improved Accuracy: 74.68%
Regularized Logistic Regression Accuracy: 76.62%

Future Improvements
Experiment with additional feature engineering techniques.
Try deep learning models for better prediction accuracy.
Expand the dataset with external medical records for improved insights.
