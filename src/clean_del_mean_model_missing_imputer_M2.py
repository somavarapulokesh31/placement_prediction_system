
import pandas as pd


# ------------------------------------------------------------
# 1. READ DATASET
# ------------------------------------------------------------

input_file = r"C:\Users\somavarapu lokesh\PycharmProjects\placement_prediction\uploads\placement_predict_50K_Raw (2) (1).csv"

output_file = r"C:\Users\somavarapu lokesh\PycharmProjects\placement_prediction\dataset\clean_del_mean_model_M2.csv"

df = pd.read_csv(input_file)

print("=" * 70)
print("ORIGINAL PLACEMENT PREDICTION DATASET")
print("=" * 70)

print(df.head())

print("\nDataset Shape:")
print(df.shape)


# ------------------------------------------------------------
# 2. CHECK MISSING VALUES
# ------------------------------------------------------------

print("\nMissing Values in Original Dataset:")
print(df.isnull().sum())


# ------------------------------------------------------------
# 3. CREATE COPY
# ------------------------------------------------------------

# The original dataset is never modified.

df_original = df.copy()


# ============================================================
# METHOD 1: DELETION
# ============================================================

# Delete rows containing any missing value.

df_deletion = df_original.dropna().copy()

print("\n" + "=" * 70)
print("1. DELETION")
print("=" * 70)

print("Original rows:", len(df_original))
print("Rows after deletion:", len(df_deletion))
print(
    "Rows deleted:",
    len(df_original) - len(df_deletion)
)


# ------------------------------------------------------------
# Create a deletion indicator
# ------------------------------------------------------------

# 1 = row contains missing value and would be deleted
# 0 = row contains no missing value

deletion_indicator = (
    df_original
    .isnull()
    .any(axis=1)
    .astype(int)
)


# ============================================================
# METHOD 2: MEAN IMPUTATION
# ============================================================

df_mean = df_original.copy()

# Find numerical columns

numeric_columns = (
    df_original
    .select_dtypes(include="number")
    .columns
    .tolist()
)

print("\nNumerical Columns:")
print(numeric_columns)


for column in numeric_columns:

    if df_mean[column].isnull().any():

        # Calculate mean excluding missing values

        mean_value = df_mean[column].mean()

        # Replace missing values with mean

        df_mean[column] = (
            df_mean[column]
            .fillna(mean_value)
        )

        print(
            "Mean used for",
            column,
            "=",
            mean_value
        )


print("\n" + "=" * 70)
print("2. MEAN IMPUTATION")
print("=" * 70)

print(df_mean.head())


# ============================================================
# METHOD 3: MEDIAN IMPUTATION
# ============================================================

df_median = df_original.copy()


for column in numeric_columns:

    if df_median[column].isnull().any():

        # Calculate median

        median_value = (
            df_median[column]
            .median()
        )

        # Replace missing values with median

        df_median[column] = (
            df_median[column]
            .fillna(median_value)
        )

        print(
            "Median used for",
            column,
            "=",
            median_value
        )


print("\n" + "=" * 70)
print("3. MEDIAN IMPUTATION")
print("=" * 70)

print(df_median.head())


# ============================================================
# METHOD 4: MODEL-BASED IMPUTATION
# ============================================================

# Pandas does not contain a machine-learning model.
#
# Therefore, a simple linear regression calculation is
# implemented using Pandas operations.

df_model = df_original.copy()

print("\n" + "=" * 70)
print("4. MODEL-BASED IMPUTATION")
print("=" * 70)


if len(numeric_columns) >= 2:

    for target_column in numeric_columns:

        # Check if target column contains missing values

        if not df_model[target_column].isnull().any():
            continue


        # ----------------------------------------------------
        # Select another numerical column as predictor
        # ----------------------------------------------------

        predictor_column = None


        for column in numeric_columns:

            if column != target_column:

                if (
                    df_model[column]
                    .notnull()
                    .sum()
                    >= 2
                ):

                    predictor_column = column
                    break


        # If predictor is not available

        if predictor_column is None:
            continue


        # ----------------------------------------------------
        # Training data
        # ----------------------------------------------------

        training_data = df_model[
            df_model[target_column].notnull()
            &
            df_model[predictor_column].notnull()
        ].copy()


        # Need at least two records

        if len(training_data) < 2:
            continue


        x = training_data[predictor_column]
        y = training_data[target_column]


        # ----------------------------------------------------
        # Calculate linear regression manually
        #
        # y = slope * x + intercept
        # ----------------------------------------------------

        x_mean = x.mean()
        y_mean = y.mean()


        numerator = (
            (x - x_mean)
            *
            (y - y_mean)
        ).sum()


        denominator = (
            (x - x_mean) ** 2
        ).sum()


        if denominator == 0:
            continue


        slope = numerator / denominator

        intercept = (
            y_mean
            -
            (slope * x_mean)
        )


        # ----------------------------------------------------
        # Find rows where target is missing
        # ----------------------------------------------------

        missing_rows = df_model[
            df_model[target_column].isnull()
            &
            df_model[predictor_column].notnull()
        ].copy()


        # ----------------------------------------------------
        # Predict missing values
        # ----------------------------------------------------

        if len(missing_rows) > 0:

            predicted_values = (
                slope
                *
                missing_rows[predictor_column]
                +
                intercept
            )


            df_model.loc[
                missing_rows.index,
                target_column
            ] = predicted_values


            print(
                "\nTarget column:",
                target_column
            )

            print(
                "Predictor column:",
                predictor_column
            )

            print(
                "Slope:",
                slope
            )

            print(
                "Intercept:",
                intercept
            )

            print(
                "Number of values predicted:",
                len(missing_rows)
            )


# ------------------------------------------------------------
# If some missing values remain, use median as fallback
# ------------------------------------------------------------

for column in numeric_columns:

    if df_model[column].isnull().any():

        median_value = (
            df_model[column]
            .median()
        )

        df_model[column] = (
            df_model[column]
            .fillna(median_value)
        )


print("\nModel-Based Imputation Result:")

print(df_model.head())


# ============================================================
# HANDLE CATEGORICAL COLUMNS FOR MODEL RESULT
# ============================================================

categorical_columns = (
    df_original
    .select_dtypes(exclude="number")
    .columns
    .tolist()
)


# Fill categorical missing values using mode

for column in categorical_columns:

    if df_model[column].isnull().any():

        mode_values = (
            df_model[column]
            .mode()
        )


        if len(mode_values) > 0:

            df_model[column] = (
                df_model[column]
                .fillna(mode_values.iloc[0])
            )


# ============================================================
# METHOD 5: MISSING INDICATOR FEATURES
# ============================================================

print("\n" + "=" * 70)
print("5. MISSING INDICATOR FEATURES")
print("=" * 70)


# Create all missing indicator columns together.
# This prevents DataFrame fragmentation.

indicator_data = {
    column + "_Missing":
        df_original[column]
        .isnull()
        .astype(int)

    for column in df_original.columns
}


indicator_df = pd.DataFrame(
    indicator_data,
    index=df_original.index
)


# Combine original data and indicator columns at once

df_indicator = pd.concat(
    [
        df_original,
        indicator_df
    ],
    axis=1
)


print("\nMissing Indicator Result:")

print(df_indicator.head())


# ============================================================
# CREATE ONE FINAL OUTPUT DATAFRAME
# ============================================================

# ------------------------------------------------------------
# A. Deletion indicator
# ------------------------------------------------------------

deletion_df = pd.DataFrame(
    {
        "Deletion_Row_Removed":
            deletion_indicator
    },
    index=df_original.index
)


# ------------------------------------------------------------
# B. Mean-imputed values
# ------------------------------------------------------------

mean_data = {
    column + "_Mean_Imputed":
        df_mean[column]

    for column in numeric_columns
}


mean_df = pd.DataFrame(
    mean_data,
    index=df_original.index
)


# ------------------------------------------------------------
# C. Median-imputed values
# ------------------------------------------------------------

median_data = {
    column + "_Median_Imputed":
        df_median[column]

    for column in numeric_columns
}


median_df = pd.DataFrame(
    median_data,
    index=df_original.index
)


# ------------------------------------------------------------
# D. Model-based imputed values
# ------------------------------------------------------------

model_data = {
    column + "_Model_Imputed":
        df_model[column]

    for column in numeric_columns
}


model_df = pd.DataFrame(
    model_data,
    index=df_original.index
)


# ------------------------------------------------------------
# E. Missing Indicator features
# ------------------------------------------------------------

indicator_data = {
    column + "_Missing":
        df_original[column]
        .isnull()
        .astype(int)

    for column in df_original.columns
}


indicator_df = pd.DataFrame(
    indicator_data,
    index=df_original.index
)


# ============================================================
# COMBINE ALL DATAFRAMES AT ONCE
# ============================================================

final_result = pd.concat(
    [
        df_original,
        deletion_df,
        mean_df,
        median_df,
        model_df,
        indicator_df
    ],
    axis=1
)


# ============================================================
# SAVE ALL RESULTS INTO ONE CSV FILE
# ============================================================

print("\n" + "=" * 70)
print("SAVING OUTPUT FILE")
print("=" * 70)


final_result.to_csv(
    output_file,
    index=False
)


print("\nCSV file created successfully!")

print("\nOutput file:")
print(output_file)


# ============================================================
# DISPLAY FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("FINAL OUTPUT")
print("=" * 70)

print(final_result.head())

print("\nFinal Dataset Shape:")
print(final_result.shape)


# ============================================================
# CHECK REMAINING MISSING VALUES
# ============================================================

print("\nMissing Values in Mean-Imputed Dataset:")

print(
    df_mean.isnull().sum()
)


print("\nMissing Values in Median-Imputed Dataset:")

print(
    df_median.isnull().sum()
)


print("\nMissing Values in Model-Imputed Dataset:")

print(
    df_model.isnull().sum()
)


# ============================================================
# VERIFY ORIGINAL DATASET IS NOT MODIFIED
# ============================================================

df_check = pd.read_csv(
    input_file
)


if df_original.equals(df_check):

    print(
        "\nOriginal dataset was NOT modified."
    )

else:

    print(
        "\nWARNING: Original dataset was modified!"
    )


# ============================================================
# COMPLETION MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("PROCESS COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nInput file:")
print(input_file)

print("\nOutput file:")
print(output_file)

print(
    "\nAll missing-values handling techniques "
    "are stored in ONE CSV file."
)
