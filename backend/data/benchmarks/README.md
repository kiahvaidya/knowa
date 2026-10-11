# Real-World Benchmark Dataset: IBM Telco Customer Churn

## 1. Overview & Provenance
- **Dataset Name**: IBM Cognos Analytics Telco Customer Churn Dataset
- **Source**: IBM Business Analytics Community / Kaggle open public archive
- **Licensing**: Public Domain / CC0 Open Data
- **File Location**: `backend/data/benchmarks/telco_churn.csv`

## 2. Dataset Dimensions & Schema
- **Total Records**: 7,043 customer accounts
- **Total Features**: 20 predictor features + 1 target (`Churn`)
- **Target Variable**: `Churn` (Binary: `"Yes"` = 1,869 accounts / 26.5%, `"No"` = 5,174 accounts / 73.5%)

### Feature Grouping:
1. **Identifiers & Demographics (Immutable Context)**:
   - `customerID`: Unique account identifier (dropped from model features to prevent memorization)
   - `gender`: Female / Male
   - `SeniorCitizen`: Binary indicator (0 / 1)
   - `Partner`: Yes / No
   - `Dependents`: Yes / No
2. **Account Information (Contextual & Intervenable)**:
   - `tenure`: Number of months the customer has stayed with the company (contextual time)
   - `Contract`: Month-to-month, One year, Two year (high-impact controllable lever)
   - `PaperlessBilling`: Yes / No
   - `PaymentMethod`: Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)
   - `MonthlyCharges`: Current monthly fee amount ($)
   - `TotalCharges`: Cumulative fee amount charged over tenure
3. **Subscribed Services (Actionable Levers)**:
   - `PhoneService`, `MultipleLines`
   - `InternetService`: DSL, Fiber optic, No
   - `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport` (valuable protective features)
   - `StreamingTV`, `StreamingMovies`

## 3. Data Preprocessing & Leakage Prevention Protocol
1. **Split Protocol**: Strict Stratified 70/15/15 train / validation / test split. Preprocessing transformers (encoders, scalers, imputers) are fit strictly on training data only.
2. **Missingness Handling**: `TotalCharges` contains 11 blank spaces for new customers (`tenure == 0`); coerced to numeric and imputed using median calculated on training split.
3. **Encoding**: Categorical strings are transformed using explicit dictionary or one-hot encodings.

## 4. Limitations & Real-World Caveats
1. **Cross-Sectional Observational Nature**: The dataset reflects an observational snapshot. Feature correlations (e.g. month-to-month contracts strongly associating with churn) reflect behavioral self-selection, not pure randomized causal effects.
2. **Unobserved Variables**: Macroeconomic competitor price shifts and localized network outages are not captured in the feature set.
