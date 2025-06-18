# Insurance Risk Analytics & Predictive Modeling

#  Final Report: Optimizing Premium Pricing and Risk Assessment for AlphaCare Insurance Solutions

## 1. Executive Summary

This report details the development of a data-driven framework to optimize premium pricing and enhance risk assessment for AlphaCare Insurance Solutions. Through extensive data preprocessing, exploratory data analysis, hypothesis testing, and advanced machine learning modeling (XGBoost for claim probability and RandomForest Regressor for claim severity), we have identified key factors influencing insurance claims.

Our models achieve an excellent ability to identify potential claims (ROC AUC of 0.8888, 88% recall for claims), allowing AlphaCare to anticipate and manage risk more effectively. Key findings indicate that **coverage type, geographic location (PostalCode/Province), and vehicle age** are the most critical determinants of claim likelihood and severity.

We recommend implementing a risk-adjusted premium strategy based on these insights to enhance profitability and ensure competitive pricing.

---

## 2. Data Overview and Preprocessing

The analysis utilized a comprehensive dataset encompassing policy details, transaction history, and claim information. Key variables included `PostalCode`, `LegalType`, `Gender`, `Province`, `CoverType`, `VehicleType`, `make`, `RegistrationYear`, `TransactionMonth`, `TotalClaims`, and `CalculatedPremiumPerTerm`.

### 2.1 Data Cleaning & Preparation

Before modeling and analysis, extensive preprocessing was conducted:

- **Missing value handling**:
  - `CustomValueEstimate` was imputed using `mmcode` and `RegistrationYear`.
  - `Gender` was enriched using the `Title` column (e.g., “Mr”, “Mrs”), improving coverage from 5% to over 60%.
- **Feature engineering**:
  - Created `VehicleAgeAtTransaction` as a key numerical feature (difference between `TransactionMonth` and `RegistrationYear`).
  - Constructed `HasClaim` as a binary target column.
- **Zero-premium policy investigation**:
  - Identified over 380,000 rows with `TotalPremium = 0`, and flagged risky records with `TotalClaims > 0` and `TotalPremium = 0`.
- **LossRatio issues**:
  - Carefully avoided division-by-zero when calculating `LossRatio`.

### 2.2 Data Version Control (DVC)

To ensure reproducibility and version tracking of data:

- **DVC (Data Version Control)** was used to track the cleaned dataset.
- The `.gitignore` was automatically updated to avoid Git tracking large files.
- Metadata files were committed to Git, and the dataset was pushed to a local DVC remote.
- This setup enabled collaboration and ensured reproducible pipelines in future experiments.

### 2.3 Preprocessing Steps (Continued)

- **Categorical Encoding**: Categorical features were one-hot encoded for model compatibility.
- **Class Imbalance**: For the `HasClaim` (binary claim indicator) target, the severe class imbalance was addressed using `scale_pos_weight` in the XGBoost model.

# Final Report: Optimizing Premium Pricing and Risk Assessment for AlphaCare Insurance Solutions

---

### **1. Executive Summary**

This report details the development of a data-driven framework to optimize premium pricing and enhance risk assessment for AlphaCare Insurance Solutions. Through extensive data preprocessing, exploratory data analysis, hypothesis testing, and advanced machine learning modeling (XGBoost for claim probability and RandomForest Regressor for claim severity), we have identified key factors influencing insurance claims. Our models achieve an excellent ability to identify potential claims (ROC AUC of 0.8888, 88% recall for claims), allowing AlphaCare to anticipate and manage risk more effectively. Key findings indicate that **coverage type, geographic location (PostalCode/Province), and vehicle age** are the most critical determinants of claim likelihood and severity. We recommend implementing a risk-adjusted premium strategy based on these insights to enhance profitability and ensure competitive pricing.

### **2. Introduction**

AlphaCare Insurance Solutions seeks to refine its premium pricing strategies and improve its understanding of underlying risk factors associated with policyholders. The current methods may not fully leverage available data, potentially leading to sub-optimal pricing and missed opportunities for risk mitigation. This project aims to address these challenges by building predictive models and providing actionable insights into claim probability and severity.

### **3. Data Overview and Preprocessing**

The analysis utilized a comprehensive dataset encompassing policy details, transaction history, and claim information. Key variables included `PostalCode`, `LegalType`, `Gender`, `Province`, `CoverType`, `VehicleType`, `make`, `RegistrationYear`, `TransactionMonth`, `TotalClaims`, and `CalculatedPremiumPerTerm`.

**Preprocessing Steps:**
* **Feature Engineering:** A crucial `VehicleAgeAtTransaction` feature was engineered by calculating the difference between `TransactionMonth` and `RegistrationYear`, proving to be highly predictive.
* **Missing Value Handling:** `Gender` was imputed to `GenderFilled`, and 'Not specified' rows in `GenderFilled` were subsequently dropped to ensure data quality.
* **Categorical Encoding:** Categorical features were one-hot encoded for model compatibility.
* **Class Imbalance:** For the `HasClaim` (binary claim indicator) target, the severe class imbalance was addressed using `scale_pos_weight` in the XGBoost model.

### **4. Exploratory Data Analysis (EDA) & Hypothesis Testing**

Initial data exploration and statistical tests revealed significant patterns:

* **Average Loss Ratio by Gender:** Male policyholders exhibited a higher average loss ratio (~0.30) compared to Female policyholders (~0.25), suggesting males may file more or costlier claims.
* **Average Loss Ratio by Province:** Urbanized provinces like **Gauteng** showed the highest loss ratio (~0.35), followed by Mpumalanga and Limpopo. Conversely, Northern Cape and Free State had the lowest ratios (~0.05–0.10). This highlights regional disparities in risk.
* **Average Loss Ratio by Vehicle Type:** Partial insights indicated differing median loss ratios between passenger and commercial vehicles, with commercial vehicles potentially having higher ratios. Further data would strengthen this observation.
* **Average Total Claims by Make (Top 10):** Luxury brands (**Audi, BMW, Mercedes-Benz**) and **Golden Journey** exhibited the highest average claim amounts (~150–175), while **Toyota** and **Hyundai** were mid-range (~75–100). This suggests higher repair costs for expensive vehicles and specific risks associated with certain makes.
* **Claim Frequency by Province:** **Gauteng** consistently led in claim frequency (~0.0035 claims per policy), aligning with its high loss ratio, whereas Northern Cape was lowest (~0.0005).
* **Monthly Claims Over Time:** Analysis revealed **seasonality** in claims (peaks during holiday months) and a possible **upward trend post-2017**, indicating evolving market conditions or portfolio growth.
* **Zero Premium Policies Over Time:** A sharp decline in zero-premium policies from nearly zero in 2013 to 35,000 in 2015 indicates a significant shift in policy distribution or business strategy.

**Hypothesis Test Results:**
* **Province vs. HasClaim:** Chi-Squared test showed a significant difference (p-value: 0.000000), confirming provincial impact on claim frequency.
* **PostalCode vs. HasClaim:** Highly significant difference (p-value: 0.000000), indicating strong correlation with claim frequency.
* **Gender vs. HasClaim:** Chi-Squared test indicated a statistically significant difference (p-value: 0.000066) in claim frequency, though its practical impact was observed to be minimal. T-test for gender vs. claim severity showed no significant difference (p-value: 0.985340).
* **Vehicle Type vs. HasClaim/Claim Severity:** No significant difference found for either claim frequency (p-value: 0.484716) or severity (p-value: 0.204881) based on vehicle type from these tests, though other analyses suggested potential differences.

### **5. Model Development and Evaluation**

We developed two primary models to address claim probability and severity, and evaluated a benchmark for premium estimation.

#### **5.1 Claim Probability Model (XGBoost Classifier)**

* **Objective:** Predict `HasClaim` (binary: 0 for no claim, 1 for claim).
* **Model:** XGBoost Classifier, chosen for its strong performance on tabular and imbalanced data. `scale_pos_weight` was used to prioritize correct identification of the minority `HasClaim=1` class (calculated at 359.23).
* **Performance:**
    * **Training Data Shape:** (798988, 86)
    * **Testing Data Shape:** (199748, 86)
    * **ROC AUC Score:** **0.8888** (Excellent discriminatory power).

* **Classification Report & Confusion Matrix:**
    | Metric           | Class 0 (No Claim) | Class 1 (Has Claim) |
    | :--------------- | :----------------- | :------------------ |
    | Precision        | 1.00               | **0.01** |
    | Recall           | 0.79               | **0.88** |
    | F1-Score         | 0.88               | 0.02                |
    | Support          | 199193             | 555                 |
    | **Accuracy** | **0.79** |                     |
    | **Macro Avg** | 0.51               | 0.84                |
    | **Weighted Avg** | 1.00               | 0.79                |

    **Confusion Matrix:**
    * **True Negatives (TN): 157,846** (Correctly predicted no claim)
    * **False Positives (FP): 41,347** (Incorrectly predicted a claim - false alarms)
    * **False Negatives (FN): 66** (Missed actual claims - critical error)
    * **True Positives (TP): 489** (Correctly identified actual claims)

* **Interpretation:** The model demonstrates **exceptional recall (0.88) for the positive class (`HasClaim=1`)**, meaning it is highly effective at identifying genuine claims and has a very low rate of missing claims (only 66 FNs). This is crucial for AlphaCare to anticipate and manage potential payouts. The trade-off is a low precision (0.01) due to a high number of false positives (41,347 FPs). This implies that while the model flags many policies as high-risk, a significant portion of these do not ultimately result in a claim. This balance is often acceptable in insurance to avoid the higher cost of missing actual claims.

#### **5.2 Claim Severity Model (RandomForest Regressor - Claimers Only)**

* **Objective:** Predict `TotalClaims` amount *for policies that have a claim*.
* **Model:** RandomForest Regressor, trained only on the subset of data where `HasClaim=1`.
* **Performance:**
    * **RMSE:** 35570.83
    * **R² Score:** **0.1115**

* **Interpretation:** While the R² score is relatively low, indicating that predicting the exact claim amount is challenging (much variance is unexplained), the model still provides a valuable *estimate* of potential claim severity. This estimation, even if imperfect, is vital for setting appropriate reserves and understanding the financial exposure of different claim types.

#### **5.3 Premium Estimation Model (RandomForest Regressor - Benchmark)**

* **Objective:** Predict `CalculatedPremiumPerTerm`.
* **Model:** RandomForest Regressor.
* **Performance:**
    * **RMSE:** 28.85
    * **R² Score:** **0.9829**

* **Interpretation:** This model's extremely high R² score indicates that it very accurately predicts the *currently calculated premium*. This serves as an excellent benchmark, showing that the features used are highly correlated with the existing pricing structure. It also validates the feature set for premium calculation.

### **6. Model Interpretability with SHAP**

SHAP (SHapley Additive exPlanations) was used to explain the predictions of both the Claim Probability (XGBoost) and Claim Severity (RandomForest Regressor) models, providing insights into feature importance and their directional impact.

#### **6.1 SHAP for Claim Probability (XGBoost)**

* **SHAP Feature Importance Plot (Mean Absolute SHAP Value):**
    * **Top Features (Most to Least Important):** `CoverType_Own Damage`, `CoverType_Windscreen`, `CoverType_Income Protector`, `CoverType_Passenger Liability`, `CoverType_Signage and Vehicle Wraps`, `CoverType_Cleaning and Removal of Accident Debris`, `PostalCode`, `CoverType_Basic Excess Waiver`, `CoverType_Keys and Alarms`, `VehicleAgeAtTransaction`, `CoverType_Emergency Charges`, `CoverType_Third Party`, `Province_Gauteng`, `make_TOYOTA`, `CoverType_Credit Protection`, `Province_North West`, `GenderFilled_Male`, `LegalType_Private company`, `Province_Mpumalanga`, `make_GOLDEN JOURNEY`.
    * **Interpretation:** **Coverage types overwhelmingly dominate the model's decisions**, indicating they are the primary drivers of claim likelihood. `PostalCode` and `VehicleAgeAtTransaction` are also highly influential.
* **SHAP Summary Plot (Directional Impact):**
    * **`CoverType_Own Damage`:** Policies with `Own Damage` coverage (high values) **strongly increase the predicted claim likelihood**. This is a direct and logical driver for the model's positive claim predictions.
    * **`VehicleAgeAtTransaction`:** Higher values for `VehicleAgeAtTransaction` (older vehicles) consistently **increase the predicted probability of a claim**.
    * **`PostalCode`:** Specific postal codes demonstrate a strong impact on predictions, reflecting varying risk profiles across geographic locations.
    * **`Province_Gauteng`:** Shows a non-linear impact, meaning its influence on claim probability varies depending on other factors or specific ranges within the province.
    * **`GenderFilled_Male`:** Has a minimal impact on predicted claim likelihood, with scattered SHAP values, confirming its minor influence despite statistical significance.

#### **6.2 SHAP for Claim Severity (RandomForest Regressor - Claimers Only)**

* **Top Features for Claim Severity:** `PostalCode`, `CoverType_Own Damage`, `Vehicle Type` (Heavy commercial being highest, followed by medium, light, passenger vehicle, then bus), `Vehicle Make` (e.g., Luxury brands correlating with higher amounts).
* **Interpretation:** For policies that *do* result in a claim, the **geographic location (`PostalCode`) and specific coverage (`CoverType_Own Damage`) are paramount in determining the payout amount.** `Vehicle Type` also plays a crucial role, with commercial vehicles, particularly heavy commercial, indicating significantly higher payout amounts, logically due to their higher repair costs and potential for greater damage. Luxury vehicle makes are also associated with higher claim amounts due to more expensive repairs.

### **7. Business Recommendations and Strategic Implications**

Based on the robust models and their interpretability, we propose the following strategic recommendations for AlphaCare:

#### **7.1 Risk-Based Premium Pricing Optimization**

* **Implement a two-stage pricing model:** Leverage the output of both the Claim Probability (XGBoost) and Claim Severity (RandomForest Regressor) models. A refined premium can be calculated as:
    `Optimal Premium = P(HasClaim) * E[Claim Amount | HasClaim] + Expense Loading + Profit Margin`
    * **`P(HasClaim)`:** Derived from the XGBoost model's predicted probability for each policyholder.
    * **`E[Claim Amount | HasClaim]`:** Derived from the RandomForest Severity model's predicted claim amount for policies likely to claim.
* **Dynamic Premium Adjustments:**
    * **Coverage-Specific Premiums:** Policies including **`Own Damage` and `Windscreen` coverage should incur higher premiums**, as these are the strongest indicators of claim likelihood and severity.
    * **Geographic Adjustments:** Implement granular pricing based on **`PostalCode`** and `Province` loss ratios. Policies in high-risk areas like **Gauteng** should have appropriately adjusted higher premiums.
    * **Vehicle Age Adjustments:** Older vehicles, identified by `VehicleAgeAtTransaction`, should be associated with higher premiums due to increased claim probability.
    * **Vehicle Make/Type-Based Adjustments:** Increase premiums for **luxury brands (Audi, BMW, Mercedes-Benz)** and **commercial vehicles (especially heavy commercial)**, given their correlation with higher claim severity.
* **Targeted Pricing for Specific Risks:** Consider bespoke pricing structures for `Golden Journey` make vehicles due to their disproportionately high average claim amounts.

#### **7.2 Enhanced Underwriting and Risk Mitigation**

* **Focus on Key Risk Factors:** Underwriters should prioritize `CoverType` combinations, `PostalCode`, `VehicleAgeAtTransaction`, `VehicleType`, and `Vehicle Make` during policy assessment, as these are the most predictive features.
* **Fraud Detection:** High-frequency claim provinces like **Gauteng** warrant enhanced fraud detection measures and stricter claim verification processes.
* **Seasonality Planning:** Utilize the identified seasonal claim trends to optimize operational staffing (e.g., claims processing, customer service) and financial reserves during peak periods.

#### **7.3 Strategic Business Planning**

* **Customer Segmentation:** Develop risk profiles for policyholders based on their predicted claim probability and severity. This can inform targeted marketing efforts (e.g., offering competitive rates to identified low-risk segments) and tailored policy offerings.
* **Product Development:** Insights from feature importance can guide the development of new coverage options or risk-reduction programs.
* **Data Collection Enhancement:** Identify gaps in current data (e.g., more granular vehicle usage, driver behavior) that could further improve model accuracy and interpretability.

### **8. Limitations and Future Work**

**8.1 Limitations:**
* **Data Granularity:** The models could benefit from more granular data on driver behavior, annual mileage, historical accident records, and detailed claim causes, which were not available.
* **Severity Model Performance:** The R² for the severity model indicates that predicting the exact claim amount remains challenging, suggesting inherent variability not fully captured by current features.
* **Thresholding:** The current high false positive rate of the probability model might require further fine-tuning of the prediction threshold based on AlphaCare's specific cost-benefit analysis for false positives versus false negatives.
* **Static Analysis:** The analysis is based on historical data up to a certain point and does not account for real-time changes unless continuously updated.

**8.2 Future Work:**
* **Dynamic Model Updates:** Implement a system for regular model retraining and deployment to adapt to evolving market conditions and claim patterns.
* **Explore Advanced Techniques:** Investigate other machine learning models (e.g., deep learning for complex interactions) or ensemble methods for potentially higher accuracy.
* **Threshold Optimization:** Conduct a detailed cost-benefit analysis to determine the optimal probability threshold for classifying a "claim" based on the business costs of false positives versus false negatives.
* **A/B Testing:** Design and execute controlled A/B tests for new premium structures in real-world scenarios to validate their impact on profitability and customer acquisition.
* **External Data Integration:** Explore integrating external datasets such as traffic data, crime statistics, or economic indicators to enhance predictive power.