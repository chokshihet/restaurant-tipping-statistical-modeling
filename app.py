import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

st.set_page_config(
    page_title="Restaurant Tipping Behavior",
    layout="wide"
)

st.title("Restaurant Tipping Behavior Analysis")
st.write("Interactive Statistical Modeling Dashboard")

df = pd.read_csv("data/tips.csv")

tab1, tab2, tab3 = st.tabs([
    "Data Exploration",
    "Hypothesis Testing Lab",
    "Live Prediction & Diagnostics"
])

with tab1:
    st.header("Data Exploration")

    st.sidebar.header("Filters")

    bill_min = float(df["total_bill"].min())
    bill_max = float(df["total_bill"].max())

    bill_range = st.sidebar.slider(
        "Select Total Bill Range",
        min_value=bill_min,
        max_value=bill_max,
        value=(bill_min, bill_max)
    )

    selected_days = st.sidebar.multiselect(
        "Select Day",
        options=df["day"].unique(),
        default=list(df["day"].unique())
    )

    selected_smoker = st.sidebar.multiselect(
        "Select Smoker Status",
        options=df["smoker"].unique(),
        default=list(df["smoker"].unique())
    )

    filtered_df = df[
        (df["total_bill"] >= bill_range[0]) &
        (df["total_bill"] <= bill_range[1]) &
        (df["day"].isin(selected_days)) &
        (df["smoker"].isin(selected_smoker))
    ]

    st.subheader("Filtered Dataset")

    st.dataframe(
        filtered_df,
        use_container_width=True
    )

    st.subheader("Summary Statistics")

    st.dataframe(
        filtered_df[
            ["total_bill", "tip", "size"]
        ].describe(),
        use_container_width=True
    )

    st.subheader("Key Metrics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Number of Records",
        len(filtered_df)
    )

    col2.metric(
        "Average Bill",
        f"${filtered_df['total_bill'].mean():.2f}"
    )

    col3.metric(
        "Average Tip",
        f"${filtered_df['tip'].mean():.2f}"
    )

    col4.metric(
        "Average Party Size",
        f"{filtered_df['size'].mean():.2f}"
    )

    st.subheader("Distribution of Total Bill")

    fig1, ax1 = plt.subplots()

    sns.histplot(
        data=filtered_df,
        x="total_bill",
        kde=True,
        ax=ax1
    )

    ax1.set_title("Distribution of Total Bill")

    st.pyplot(fig1)

    st.subheader("Distribution of Tips")

    fig2, ax2 = plt.subplots()

    sns.histplot(
        data=filtered_df,
        x="tip",
        kde=True,
        ax=ax2
    )

    ax2.set_title("Distribution of Tips")

    st.pyplot(fig2)

    st.subheader("Total Bill vs Tip")

    fig3, ax3 = plt.subplots()

    sns.scatterplot(
        data=filtered_df,
        x="total_bill",
        y="tip",
        hue="smoker",
        ax=ax3
    )

    ax3.set_title("Total Bill vs Tip")

    st.pyplot(fig3)

    st.subheader("Correlation Matrix")

    correlation_matrix = filtered_df[
        ["total_bill", "tip", "size"]
    ].corr()

    fig4, ax4 = plt.subplots()

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        ax=ax4
    )

    st.pyplot(fig4)

with tab2:
    st.header("Hypothesis Testing Lab")

    st.write(
        "Select a categorical variable and a numerical variable "
        "to perform a statistical hypothesis test."
    )

    categorical_variable = st.selectbox(
        "Select Categorical Variable",
        ["smoker", "sex", "day", "time"]
    )

    numerical_variable = st.selectbox(
        "Select Numerical Variable",
        ["tip", "total_bill", "size"]
    )

    alpha = 0.05

    groups = df[categorical_variable].dropna().unique()

    st.subheader("Selected Test")

    # -------------------------------------------------
    # TWO GROUPS
    # -------------------------------------------------

    if len(groups) == 2:

        group1_name = groups[0]
        group2_name = groups[1]

        group1 = df[
            df[categorical_variable] == group1_name
        ][numerical_variable]

        group2 = df[
            df[categorical_variable] == group2_name
        ][numerical_variable]

        st.write(
            f"Comparing **{group1_name}** vs **{group2_name}** "
            f"for **{numerical_variable}**"
        )

        # ---------------------------------------------
        # Hypotheses
        # ---------------------------------------------

        st.subheader("Hypotheses")

        st.write(
            f"**H₀:** There is no significant difference in "
            f"{numerical_variable} between {group1_name} and {group2_name}."
        )

        st.write(
            f"**H₁:** There is a significant difference in "
            f"{numerical_variable} between {group1_name} and {group2_name}."
        )

        st.write("Significance Level: α = 0.05")

        # ---------------------------------------------
        # Shapiro-Wilk Test
        # ---------------------------------------------

        st.subheader("1. Normality Test — Shapiro-Wilk")

        shapiro1 = stats.shapiro(group1)
        shapiro2 = stats.shapiro(group2)

        col1, col2 = st.columns(2)

        with col1:
            st.write(f"### {group1_name}")
            st.write(
                f"Statistic: {shapiro1.statistic:.4f}"
            )
            st.write(
                f"p-value: {shapiro1.pvalue:.4f}"
            )

            if shapiro1.pvalue > alpha:
                st.success("Data is approximately normal")
            else:
                st.warning("Data is not normally distributed")

        with col2:
            st.write(f"### {group2_name}")
            st.write(
                f"Statistic: {shapiro2.statistic:.4f}"
            )
            st.write(
                f"p-value: {shapiro2.pvalue:.4f}"
            )

            if shapiro2.pvalue > alpha:
                st.success("Data is approximately normal")
            else:
                st.warning("Data is not normally distributed")

        # ---------------------------------------------
        # Levene Test
        # ---------------------------------------------

        st.subheader("2. Equal Variance Test — Levene's Test")

        levene_result = stats.levene(
            group1,
            group2
        )

        st.write(
            f"Statistic: {levene_result.statistic:.4f}"
        )

        st.write(
            f"p-value: {levene_result.pvalue:.4f}"
        )

        if levene_result.pvalue > alpha:
            st.success("Equal variance assumption is satisfied")
        else:
            st.warning("Equal variance assumption is not satisfied")

        # ---------------------------------------------
        # Choose Statistical Test
        # ---------------------------------------------

        st.subheader("3. Final Statistical Test")

        if (
            shapiro1.pvalue > alpha
            and shapiro2.pvalue > alpha
        ):

            equal_variance = (
                levene_result.pvalue > alpha
            )

            test_result = stats.ttest_ind(
                group1,
                group2,
                equal_var=equal_variance
            )

            test_name = "Independent Two-Sample t-test"

        else:

            test_result = stats.mannwhitneyu(
                group1,
                group2,
                alternative="two-sided"
            )

            test_name = "Mann-Whitney U Test"

        st.write(
            f"**Test Used:** {test_name}"
        )

        st.write(
            f"**Test Statistic:** {test_result.statistic:.4f}"
        )

        st.write(
            f"**p-value:** {test_result.pvalue:.4f}"
        )

        # ---------------------------------------------
        # Conclusion
        # ---------------------------------------------

        st.subheader("Conclusion")

        if test_result.pvalue < alpha:

            st.error(
                "Reject H₀: There is a statistically significant "
                "difference between the two groups."
            )

        else:

            st.success(
                "Fail to Reject H₀: There is not enough evidence "
                "to conclude that the groups are significantly different."
            )

        # ---------------------------------------------
        # Group Summary
        # ---------------------------------------------

        st.subheader("Group Summary")

        summary = df.groupby(
            categorical_variable
        )[numerical_variable].agg(
            ["count", "mean", "median", "std"]
        )

        st.dataframe(
            summary,
            use_container_width=True
        )

        # ---------------------------------------------
        # Box Plot
        # ---------------------------------------------

        st.subheader("Group Comparison")

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        sns.boxplot(
            data=df,
            x=categorical_variable,
            y=numerical_variable,
            ax=ax
        )

        ax.set_title(
            f"{numerical_variable} by {categorical_variable}"
        )

        st.pyplot(fig)

    # -------------------------------------------------
    # THREE OR MORE GROUPS
    # -------------------------------------------------

    else:

        st.write(
            f"Since **{categorical_variable}** contains "
            f"{len(groups)} groups, a One-Way ANOVA will be used."
        )

        st.subheader("Hypotheses")

        st.write(
            f"**H₀:** The mean {numerical_variable} is equal "
            f"across all {categorical_variable} groups."
        )

        st.write(
            f"**H₁:** At least one group has a significantly "
            f"different mean {numerical_variable}."
        )

        st.write("Significance Level: α = 0.05")

        group_data = [
            df[
                df[categorical_variable] == group
            ][numerical_variable]
            for group in groups
        ]

        anova_result = stats.f_oneway(
            *group_data
        )

        st.subheader("One-Way ANOVA Results")

        st.write(
            f"**F-Statistic:** {anova_result.statistic:.4f}"
        )

        st.write(
            f"**p-value:** {anova_result.pvalue:.4f}"
        )

        st.subheader("Conclusion")

        if anova_result.pvalue < alpha:

            st.error(
                "Reject H₀: At least one group has a "
                "significantly different mean."
            )

        else:

            st.success(
                "Fail to Reject H₀: There is not enough evidence "
                "to conclude that the group means are different."
            )

        # ---------------------------------------------
        # Group Summary
        # ---------------------------------------------

        st.subheader("Group Summary")

        summary = df.groupby(
            categorical_variable
        )[numerical_variable].agg(
            ["count", "mean", "median", "std"]
        )

        st.dataframe(
            summary,
            use_container_width=True
        )

        # ---------------------------------------------
        # Box Plot
        # ---------------------------------------------

        st.subheader("Group Comparison")

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        sns.boxplot(
            data=df,
            x=categorical_variable,
            y=numerical_variable,
            ax=ax
        )

        ax.set_title(
            f"{numerical_variable} by {categorical_variable}"
        )

        st.pyplot(fig)

with tab3:
    st.header("Live Prediction & Diagnostics")

    st.write(
        "Enter restaurant details below to generate a predicted tip "
        "using the multiple linear regression model."
    )

    # -------------------------------------------------
    # Prepare regression data
    # -------------------------------------------------

    model_df = pd.get_dummies(
        df[
            [
                "tip",
                "total_bill",
                "size",
                "sex",
                "smoker",
                "time"
            ]
        ],
        drop_first=True,
        dtype=int
    )

    X = model_df.drop("tip", axis=1)
    y = model_df["tip"]

    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()

    # -------------------------------------------------
    # User Inputs
    # -------------------------------------------------

    st.subheader("Enter Customer Details")

    col1, col2 = st.columns(2)

    with col1:

        total_bill_input = st.number_input(
            "Total Bill",
            min_value=float(df["total_bill"].min()),
            max_value=float(df["total_bill"].max()),
            value=20.00,
            step=1.00
        )

        size_input = st.number_input(
            "Party Size",
            min_value=int(df["size"].min()),
            max_value=int(df["size"].max()),
            value=2,
            step=1
        )

    with col2:

        sex_input = st.selectbox(
            "Sex",
            options=df["sex"].unique(),
            key="prediction_sex"
        )

        smoker_input = st.selectbox(
            "Smoker",
            options=df["smoker"].unique(),
            key="prediction_smoker"
        )

        time_input = st.selectbox(
            "Time",
            options=df["time"].unique(),
            key="prediction_time"
        )

    # -------------------------------------------------
    # Prepare New Customer Data
    # -------------------------------------------------

    new_customer = pd.DataFrame({
        "total_bill": [total_bill_input],
        "size": [size_input],
        "sex": [sex_input],
        "smoker": [smoker_input],
        "time": [time_input]
    })

    # Convert categorical variables to dummy variables
    new_customer = pd.get_dummies(
        new_customer,
        columns=["sex", "smoker", "time"],
        dtype=int
    )

    # Predictor columns used by model, excluding intercept
    predictor_columns = [
        col for col in X.columns
        if col != "const"
    ]

    # Add any columns missing from the new observation
    for col in predictor_columns:
        if col not in new_customer.columns:
            new_customer[col] = 0

    # Remove any columns not used by the model
    new_customer = new_customer.reindex(
        columns=predictor_columns,
        fill_value=0
    )

    # Add intercept
    prediction_data = sm.add_constant(
        new_customer,
        has_constant="add"
    )

    # Ensure exact same column order as regression model
    prediction_data = prediction_data.reindex(
        columns=X.columns,
        fill_value=0
    )

    # -------------------------------------------------
    # Generate Prediction
    # -------------------------------------------------

    prediction = model.get_prediction(
        prediction_data
    )

    prediction_summary = prediction.summary_frame(
        alpha=0.05
    )

    predicted_tip = prediction_summary[
        "mean"
    ].iloc[0]

    confidence_lower = prediction_summary[
        "mean_ci_lower"
    ].iloc[0]

    confidence_upper = prediction_summary[
        "mean_ci_upper"
    ].iloc[0]

    prediction_lower = prediction_summary[
        "obs_ci_lower"
    ].iloc[0]

    prediction_upper = prediction_summary[
        "obs_ci_upper"
    ].iloc[0]

    # -------------------------------------------------
    # Prediction Result
    # -------------------------------------------------

    st.subheader("Prediction Result")

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric(
        "Predicted Tip",
        f"${predicted_tip:.2f}"
    )

    metric2.metric(
        "95% Confidence Interval",
        f"${confidence_lower:.2f} - ${confidence_upper:.2f}"
    )

    metric3.metric(
        "95% Prediction Interval",
        f"${prediction_lower:.2f} - ${prediction_upper:.2f}"
    )

    st.divider()

    # -------------------------------------------------
    # Model Performance
    # -------------------------------------------------

    st.subheader("Model Performance")

    model_col1, model_col2 = st.columns(2)

    model_col1.metric(
        "R-squared",
        f"{model.rsquared:.3f}"
    )

    model_col2.metric(
        "Adjusted R-squared",
        f"{model.rsquared_adj:.3f}"
    )

    # -------------------------------------------------
    # Regression Coefficients
    # -------------------------------------------------

    st.subheader("Regression Coefficients")

    coefficient_table = pd.DataFrame({
        "Coefficient": model.params,
        "p-value": model.pvalues,
        "CI Lower": model.conf_int()[0],
        "CI Upper": model.conf_int()[1]
    })

    st.dataframe(
        coefficient_table.round(4),
        use_container_width=True
    )

    st.divider()

    # -------------------------------------------------
    # Residual Diagnostics
    # -------------------------------------------------

    st.subheader("Residual Diagnostic Plots")

    fitted_values = model.fittedvalues
    residuals = model.resid

    # Residuals vs Fitted Plot
    st.write("### Residuals vs Fitted Values")

    fig1, ax1 = plt.subplots(
        figsize=(8, 5)
    )

    sns.scatterplot(
        x=fitted_values,
        y=residuals,
        ax=ax1
    )

    ax1.axhline(
        y=0,
        linestyle="--"
    )

    ax1.set_xlabel(
        "Fitted Values"
    )

    ax1.set_ylabel(
        "Residuals"
    )

    ax1.set_title(
        "Residuals vs Fitted Values"
    )

    st.pyplot(fig1)

    # -------------------------------------------------
    # Q-Q Plot
    # -------------------------------------------------

    st.write("### Q-Q Plot of Residuals")

    fig2 = sm.qqplot(
        residuals,
        line="45",
        fit=True
    )

    plt.title(
        "Q-Q Plot of Residuals"
    )

    st.pyplot(fig2)

    # -------------------------------------------------
    # Jarque-Bera Test
    # -------------------------------------------------

    st.subheader("Residual Normality Test")

    jb_result = stats.jarque_bera(
        residuals
    )

    st.write(
        f"**Jarque-Bera Statistic:** "
        f"{jb_result.statistic:.4f}"
    )

    st.write(
        f"**p-value:** "
        f"{jb_result.pvalue:.4f}"
    )

    if jb_result.pvalue > 0.05:

        st.success(
            "Fail to Reject H₀: Residuals are approximately normal."
        )

    else:

        st.warning(
            "Reject H₀: Residuals show evidence of non-normality."
        )

    # -------------------------------------------------
    # VIF
    # -------------------------------------------------

    st.subheader(
        "Multicollinearity Check — Variance Inflation Factor"
    )

    continuous_predictors = df[
        ["total_bill", "size"]
    ].copy()

    continuous_predictors = sm.add_constant(
        continuous_predictors
    )

    vif_data = pd.DataFrame()

    vif_data["Variable"] = (
        continuous_predictors.columns
    )

    vif_data["VIF"] = [
        variance_inflation_factor(
            continuous_predictors.values,
            i
        )
        for i in range(
            continuous_predictors.shape[1]
        )
    ]

    st.dataframe(
        vif_data.round(3),
        use_container_width=True
    )