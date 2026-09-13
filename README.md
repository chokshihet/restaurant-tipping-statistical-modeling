# Restaurant Tipping Behavior Analysis

## M.Sc. Data Science — Semester 1
### Lab 4: Applied Statistical Modeling & Interactive Web Dashboard

## Live Dashboard

**Streamlit Application:** [Open Live Dashboard](ADD_YOUR_STREAMLIT_URL_HERE)

> The live dashboard URL will be added after deployment to Streamlit Community Cloud.

---

## Project Overview

This project analyzes **restaurant tipping behavior** using the Tips dataset.

The project applies statistical modeling techniques to understand the relationship between restaurant bills, customer characteristics, and tip amounts. It includes exploratory data analysis, hypothesis testing, multiple linear regression, regression diagnostics, and an interactive Streamlit dashboard.

The dashboard allows users to explore the dataset, dynamically perform statistical tests, and generate real-time tip predictions.

---

## Dataset Summary

The project uses the **Restaurant Tipping Behavior (Tips) dataset**.

The dataset contains information about restaurant bills, tips, customer characteristics, meal time, and party size.

### Variables

| Variable | Description |
|---|---|
| `total_bill` | Total restaurant bill amount |
| `tip` | Tip amount given by the customer |
| `sex` | Sex of the customer |
| `smoker` | Whether the customer is a smoker |
| `day` | Day of the restaurant visit |
| `time` | Lunch or Dinner |
| `size` | Number of people in the party |

The dataset is stored locally at:

```text
data/tips.csv
```

---

## Project Objectives

The main objectives of this project are:

- Perform Exploratory Data Analysis (EDA)
- Calculate descriptive statistical measures
- Study distributions of numerical variables
- Examine relationships and correlations between variables
- Perform statistical hypothesis testing
- Build a Multiple Linear Regression model
- Evaluate regression coefficients and model performance
- Perform regression diagnostic tests
- Check multicollinearity using VIF
- Develop an interactive Streamlit dashboard
- Generate real-time tip predictions with confidence and prediction intervals

---

# Part 1: Exploratory Data Analysis

The exploratory analysis is performed in `analysis.ipynb`.

The following descriptive measures are calculated for numerical variables:

- Mean
- Median
- Standard Deviation
- Interquartile Range (IQR)
- Skewness
- Kurtosis

The main numerical variables analyzed are:

- `total_bill`
- `tip`
- `size`

Distribution analysis is performed using:

- Histograms
- KDE plots

Relationships between variables are studied using:

- Scatter plots
- Correlation matrix
- Heatmap

---

# Hypothesis Test 1: Smokers vs Non-Smokers

The first hypothesis test investigates whether tip amounts differ significantly between smokers and non-smokers.

## Hypotheses

**Null Hypothesis (H0):**

There is no significant difference in tip amounts between smokers and non-smokers.

**Alternative Hypothesis (H1):**

There is a significant difference in tip amounts between smokers and non-smokers.

**Significance Level:**

```text
α = 0.05
```

## Testing Procedure

The following procedure is used:

1. Shapiro-Wilk test is performed to check normality for both groups.
2. Levene's test is performed to check equality of variances.
3. If both groups satisfy the normality assumption, an Independent Two-Sample t-test is used.
4. If the normality assumption is not satisfied, a Mann-Whitney U test is used.
5. The final conclusion is determined using a significance level of 0.05.

---

# Hypothesis Test 2: Tips Across Days

The second hypothesis test investigates whether the average tip amount differs across restaurant days.

## Hypotheses

**Null Hypothesis (H0):**

The mean tip amount is equal across all days.

**Alternative Hypothesis (H1):**

At least one day has a significantly different mean tip amount.

**Significance Level:**

```text
α = 0.05
```

A **One-Way ANOVA** is performed to compare tip amounts across the different day groups.

---

# Part 2: Multiple Linear Regression

An **Ordinary Least Squares (OLS) Multiple Linear Regression** model is developed to predict restaurant tip amounts.

## Dependent Variable

```text
tip
```

## Predictor Variables

The following variables are used as predictors:

- `total_bill`
- `size`
- `sex`
- `smoker`
- `time`

Categorical variables are converted into dummy variables before fitting the regression model.

The regression model is fitted using:

```python
statsmodels.api.OLS
```

---

## Model Evaluation

The regression model reports:

- Regression coefficients
- p-values
- 95% confidence intervals
- R-squared
- Adjusted R-squared

These measures are used to evaluate the relationship between the predictors and tip amount and to assess the overall explanatory performance of the model.

---

# Regression Diagnostics

Several diagnostic techniques are used to evaluate the assumptions of the regression model.

## Residuals vs Fitted Values

A Residuals vs Fitted plot is used to examine:

- Linearity
- Residual patterns
- Homoscedasticity

## Q-Q Plot

A Q-Q plot is used to visually examine whether the regression residuals approximately follow a normal distribution.

## Jarque-Bera Test

The Jarque-Bera test is used to statistically evaluate residual normality.

## Variance Inflation Factor (VIF)

Variance Inflation Factor is calculated for continuous predictors to examine potential multicollinearity.

---

# Concise Synthesis of Statistical Findings

The exploratory analysis shows that restaurant tip amounts vary across observations and are associated with characteristics such as the total bill and party size. Distribution plots, scatter plots, and correlation analysis are used to understand these relationships.

For the first hypothesis test, tips from **smokers and non-smokers** are compared. Normality is first evaluated using the Shapiro-Wilk test and equality of variance using Levene's test. Based on these assumption checks, either the Independent Two-Sample t-test or Mann-Whitney U test is selected. The final statistical conclusion is determined at the **5% significance level (α = 0.05)**.

The second hypothesis test uses **One-Way ANOVA** to determine whether average tip amounts differ significantly across restaurant days. The ANOVA result is interpreted using the same significance level of 0.05.

The Multiple Linear Regression model predicts tip amount using **total bill, party size, sex, smoker status, and meal time**. Model performance is evaluated using R-squared and Adjusted R-squared, while individual predictors are evaluated using their regression coefficients, p-values, and 95% confidence intervals.

Regression assumptions are further examined using the **Residuals vs Fitted plot, Q-Q plot, Jarque-Bera test, and Variance Inflation Factor (VIF)**. Together, these analyses provide a statistical framework for understanding restaurant tipping behavior and predicting tip amounts.

> **Note:** Exact test statistics, p-values, R-squared values, and other numerical results are generated by `analysis.ipynb` and displayed in the Streamlit dashboard.

---

# Part 3: Interactive Streamlit Dashboard

The project contains an interactive Streamlit dashboard with three tabs.

## Tab 1 — Data Exploration

The Data Exploration tab provides interactive filters for:

- Total Bill Range
- Day
- Smoker Status

It dynamically displays:

- Filtered dataset
- Summary statistics
- Number of records
- Average bill
- Average tip
- Average party size
- Distribution of total bills
- Distribution of tips
- Total Bill vs Tip scatter plot
- Correlation matrix

---

## Tab 2 — Hypothesis Testing Lab

The Hypothesis Testing Lab allows users to dynamically select:

- A categorical variable
- A numerical variable

For categorical variables containing two groups, the dashboard performs:

- Shapiro-Wilk Normality Test
- Levene's Test
- Independent Two-Sample t-test or Mann-Whitney U Test

For categorical variables containing more than two groups, the dashboard performs:

- One-Way ANOVA

The dashboard dynamically displays:

- Null and alternative hypotheses
- Test used
- Test statistic
- p-value
- Statistical conclusion
- Group summary statistics
- Comparison box plot

---

## Tab 3 — Live Prediction & Diagnostics

The Live Prediction & Diagnostics tab allows the user to enter:

- Total Bill
- Party Size
- Sex
- Smoker Status
- Meal Time

Based on the selected values, the regression model generates:

- Predicted Tip
- 95% Confidence Interval
- 95% Prediction Interval

The tab also displays:

- R-squared
- Adjusted R-squared
- Regression coefficients
- Predictor p-values
- 95% coefficient confidence intervals
- Residuals vs Fitted plot
- Q-Q plot
- Jarque-Bera test
- VIF results

---

# Project Structure

```text
restaurant_tipping_project/
│
├── data/
│   └── tips.csv
│
├── analysis.ipynb
├── app.py
├── requirements.txt
└── README.md
```

---

# Technologies and Libraries Used

The project is developed using:

- Python
- Pandas
- NumPy
- SciPy
- Seaborn
- Matplotlib
- Statsmodels
- Streamlit
- Plotly
- Jupyter Notebook
- Visual Studio Code

---

# Installation and Setup

## 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Move into the project directory:

```bash
cd restaurant_tipping_project
```

## 2. Create a Virtual Environment

```bash
python -m venv venv
```

### Windows

Activate the virtual environment using:

```bash
venv\Scripts\activate
```

### macOS/Linux

Activate using:

```bash
source venv/bin/activate
```

## 3. Install Required Libraries

Install all project dependencies using:

```bash
python -m pip install -r requirements.txt
```

---

# Run the Streamlit Application

Run the following command from the project directory:

```bash
python -m streamlit run app.py
```

Streamlit will start the application and open the dashboard in the browser.

The local application is normally available at:

```text
http://localhost:8501
```

---

# Run the Statistical Analysis

Open:

```text
analysis.ipynb
```

using VS Code or Jupyter Notebook.

Run all cells sequentially from top to bottom.

The notebook contains:

- Dataset loading
- Dataset overview
- Descriptive statistics
- Distribution analysis
- Correlation analysis
- Hypothesis Test 1
- Hypothesis Test 2
- Multiple Linear Regression
- Regression coefficient analysis
- Residual diagnostics
- Jarque-Bera test
- VIF analysis

---

# Requirements

The `requirements.txt` file contains:

```text
pandas
numpy
scipy
seaborn
matplotlib
statsmodels
streamlit
plotly
```

---

# Streamlit Community Cloud Deployment

The completed dashboard can be deployed to **Streamlit Community Cloud**.

After deployment, the public application URL should be added to the **Live Dashboard** section at the top of this README.

Example:

```markdown
## Live Dashboard

**Streamlit Application:** [Open Live Dashboard](YOUR_STREAMLIT_PUBLIC_URL)
```

---

# Conclusion

This project demonstrates an end-to-end statistical modeling workflow using restaurant tipping data.

It combines **Exploratory Data Analysis, descriptive statistics, hypothesis testing, multiple linear regression, regression diagnostics, and interactive visualization**.

The Streamlit dashboard extends the statistical analysis by allowing users to interactively explore restaurant data, perform dynamic hypothesis tests, and generate real-time tip predictions with confidence and prediction intervals.
