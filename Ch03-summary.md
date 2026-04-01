# Chapter 3: Linear Regression — Lab Summary

This lab (`Ch03-linreg-lab.ipynb`) covers practical linear regression using Python's `statsmodels` library and ISLP utilities.

## Datasets

- **Boston**: 506 neighborhoods, 13 predictors. Response variable: `medv` (median house value).
- **Carseats**: 400 locations with car seat sales data. Response: `Sales`. Includes a qualitative predictor `ShelveLoc` (Bad / Medium / Good).

## Topics Covered

### 1. Simple Linear Regression
Fit a model with a single predictor (`medv ~ lstat`) using `ModelSpec` to build the design matrix and `sm.OLS` to fit the model. Demonstrates extracting coefficients, R², RSE, and plotting the fitted line with a custom `abline()` function.

### 2. Predictions and Confidence Intervals
Use `results.get_prediction()` to obtain point estimates, confidence intervals (for the mean), and prediction intervals (for individual observations) at new values of the predictor.

### 3. Diagnostic Plots
- **Residuals vs. fitted values**: detect non-linearity.
- **Leverage statistics**: identify high-influence observations via the hat matrix diagonal (`get_influence().hat_matrix_diag`).

### 4. Multiple Linear Regression
Extend to multiple predictors by passing a list to `ModelSpec`. Also shows how to include all variables at once (`Boston.columns.drop('medv')`).

### 5. Multicollinearity — Variance Inflation Factors (VIF)
Compute VIF for each predictor using `variance_inflation_factor()` to detect collinearity among predictors.

### 6. Non-linear Transformations
- Add polynomial terms with `poly('lstat', degree=2)`.
- Compare nested models with `anova_lm()`: the quadratic term is highly significant (F ≈ 177, p ≈ 0), and residual plots confirm improved fit.

### 7. Qualitative Predictors and Interaction Terms
- `ModelSpec` automatically one-hot encodes categorical variables (e.g., `ShelveLoc` → dummy variables with "Bad" as baseline).
- Interaction terms are specified as tuples, e.g., `('Income', 'Advertising')`.

## Key Libraries and Functions

| Tool | Purpose |
|------|---------|
| `statsmodels.api.OLS` | Fit OLS regression |
| `ISLP.ModelSpec (MS)` | Build design matrices (handles dummies, interactions, polynomials) |
| `ISLP.poly` | Orthogonal polynomial basis |
| `ISLP.load_data` | Load ISLP datasets |
| `results.summary()` | Full regression table |
| `results.get_prediction()` | Predictions + confidence/prediction intervals |
| `results.get_influence()` | Leverage and influence statistics |
| `anova_lm()` | Compare nested models |
| `variance_inflation_factor()` | Multicollinearity diagnostics |

## Key Results

- A simple linear regression of `medv` on `lstat` shows a significant negative relationship: higher proportions of low-income households are associated with lower median house values.
- Adding a quadratic `lstat` term significantly improves fit, removing the non-linear pattern in residuals.
- In the Carseats data, `ShelveLoc` (Good vs. Bad) has a large positive effect on sales, and the `Income:Advertising` and `Price:Age` interactions are both informative.
