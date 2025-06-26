# 🏦 Credit Risk Modeling with XGBoost

This project builds a credit risk scoring model to predict the probability of default for loan applicants. Using the "Give Me Some Credit" dataset from Kaggle, we simulate a real-world credit scoring pipeline — from preprocessing and feature engineering to model training, evaluation, and interpretability.

---

## 🔍 Problem Statement

Financial institutions need to assess the likelihood that a borrower will default on a loan. Misclassifying risky borrowers can result in financial losses, while rejecting creditworthy applicants can impact revenue and customer experience.

This project aims to:
- Predict loan default (binary classification)
- Balance sensitivity (catching defaulters) and specificity (not rejecting good customers)
- Provide model interpretability using SHAP
- Assign risk tiers for decision support (approve / manual review / reject)

---

## 📦 Dataset

- **Source**: [Give Me Some Credit – Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit)
- **Target variable**: `SeriousDlqin2yrs` (1 = defaulted, 0 = non-default)
- **Features include**: revolving credit utilization, number of delinquencies, debt ratio, etc.

---

## 🔧 Pipeline Overview

1. **Data Cleaning & Preprocessing**
   - Imputation using median strategy
   - Weight of Evidence (WOE) encoding for interpretability
   - Feature scaling (StandardScaler)
   
2. **Exploratory Data Analysis**
   - Class imbalance visualization
   - Distribution plots of key features

3. **Modeling**
   - Logistic Regression (baseline)
   - XGBoost with `scale_pos_weight` to handle class imbalance
   - Threshold tuning for better recall vs precision trade-off

4. **Evaluation**
   - AUC, F1-Score, Confusion Matrix, Precision-Recall Curve
   - Custom threshold selection based on business needs

5. **Interpretability**
   - SHAP summary plots to show global feature importance
   - SHAP waterfall plot to explain individual predictions

6. **Risk Tiers**
   - Probabilistic classification into:
     - Low Risk (Approve)
     - Medium Risk (Manual Review)
     - High Risk (Reject)

---

## 📊 Visualizations to Include

Be sure to include the following plots in your README or as `images/`:

1. **Class Imbalance Plot**  
   Bar chart of 0s vs 1s in the target variable  
   `figures/class_distribution.png`

2. **Precision-Recall Curve**  
   Shows the trade-off for different thresholds  
   ![ROC Curve](figures/roc_curve.png)

3. **Confusion Matrix (Annotated)**  
   For default vs non-default predictions  
   ![Confusion Matrix LR](figures/cm_lr.png)

   ![Confusion Matrix XGBoost](figures/cm_xgb.png)

5. **SHAP Summary Plot (Bar or Beeswarm)**  
   Top predictive features across the test set  
   ![SHAP Summary](figures/shap_summary_bar.png)

6. **SHAP Waterfall Plot (Single Example)**  
   Explanation for why a high-risk prediction was made  
   ![SHAP Waterfall Plot](figures/shap_waterfall_high_risk.png)

7. **Risk Tier Distribution Plot**  
   Pie chart or bar plot showing distribution across Approve / Review / Reject  
   `figures/risk_tiers.png`

---

## 🧠 Key Results

| Metric                  | Value |
|-------------------------|-------|
| Accuracy                | 92%   |
| Recall (Defaulters)     | 51%   |
| Precision (Defaulters)  | 40%   |
| AUC Score               | 0.79  |
| Risk Tier Assignment    | ✓ Implemented |

---

## 🚀 Future Work

- Hyperparameter tuning via Optuna or GridSearchCV
- Ensemble modeling (stacking Logistic + XGBoost)
- Deploy as a Streamlit app for business users
