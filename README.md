
# Diabetes Data Analysis & Prediction

An end-to-end data analysis and machine learning project focused on identifying patterns and predicting diabetes using clinical data.

---

## Project Overview

This project performs a comprehensive analysis of a diabetes dataset, combining:

- Data cleaning and preprocessing  
- Exploratory data analysis (EDA)  
- Dimensionality reduction  
- Machine learning model building  

The objective is to **extract meaningful insights** and **build predictive models** to assist in early diabetes detection.

---

##  Key Features

### Data Cleaning
- Replaced invalid zero values with median values for relevant features  
- Ensured dataset consistency and reliability  

### Exploratory Data Analysis
- Correlation heatmap for feature relationships  
- Distribution analysis of key variables  
- Identification of important predictors  

### Dimensionality Reduction
- PCA (Principal Component Analysis)  
- t-SNE visualization  
- UMAP for high-dimensional data representation  

### Unsupervised Learning
- K-Means clustering to identify hidden patient groups  

### Supervised Learning Models
- Logistic Regression  
- Random Forest Classifier  
- XGBoost Classifier  

### Model Optimization
- Cross-validation for performance evaluation  
- GridSearchCV for hyperparameter tuning  
- Feature selection using Random Forest  

### Regularization
- L2 Regularization applied to Logistic Regression to reduce overfitting  

---

## Results

| Model                          | Accuracy |
|--------------------------------|----------|
| Logistic Regression            | 76.62%   |
| Random Forest                  | 76.62%   |
| XGBoost                        | 73.38%   |
| Optimized Random Forest (CV)   | 78.02%   |
| Feature Selection Model        | 74.68%   |

---

## Dataset

- File: `diabetes.csv`  
- Contains medical attributes such as:
  - Glucose level  
  - BMI  
  - Age  
  - Blood pressure  

---

## Installation & Setup
pip install pandas numpy seaborn matplotlib scikit-learn umap-learn xgboost

### Clone the repository
```bash
git clone https://github.com/rutujdv/DiabetesDataAnalysis.git
cd DiabetesDataAnalysis
