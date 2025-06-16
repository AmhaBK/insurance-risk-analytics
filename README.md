# Insurance Risk Analytics & Predictive Modeling

This project is part of the 10 Academy Week 3 Challenge. The objective is to analyze real car insurance data and build risk and pricing models to help AlphaCare Insurance Solutions identify low-risk customer segments and optimize premiums.


## Objectives

- Analyze historical car insurance data from South Africa.
- Identify segments with high or low insurance risk.
- Perform statistical testing to validate risk-based hypotheses.
- Build predictive models for claim severity and premium optimization.
- Enable data versioning and experiment reproducibility using DVC.


## Key Metrics

- **Loss Ratio** = TotalClaims / TotalPremium
- **Claim Severity** = Avg. claims (given a claim occurred)
- **Claim Frequency** = % of policies with ≥1 claim
- **Margin** = TotalPremium – TotalClaims



### EDA & Statistical Summary
- Cleaned and preprocessed raw `.txt` data
- Handled missing values using group-based imputation
- Created derived features (e.g., `VehicleAgeAtTransaction`)
- Flagged risky policies (claims with zero premium)
- Visualized loss ratios, claim trends, outliers, and correlations

### Data Version Control (DVC)
- Initialized DVC and set up local remote
- Tracked cleaned datasets using `dvc add`
- Committed and pushed data to remote
- Version-controlled all `.dvc` metadata via Git


## Insights

- Loss ratios vary significantly by province and vehicle type.
- Over 38% of policies have zero premiums; only 147 have claims without revenue.
- Most policies (99.3%) had no claims — consistent with real-world insurance patterns.
- Older vehicles tend to have higher average claim amounts.


## Tools & Technologies

- Python (Pandas, NumPy, Seaborn, Scikit-learn)
- Jupyter Notebooks
- Git & GitHub
- DVC (Data Version Control)



## Next Steps

- Hypothesis testing for segmentation strategies (Task 3)
- Predictive modeling of claim severity and premium pricing (Task 4)
- Model explainability via SHAP or LIME
- Final report and business recommendations


## Installation

```bash
pip install -r requirements.txt
```
