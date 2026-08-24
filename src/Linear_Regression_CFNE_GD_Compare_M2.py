# ============================================================
# LINEAR REGRESSION
# Closed-Form Normal Equation vs Gradient Descent
#
# Images are stored in ONE separate folder
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. FILE PATHS
# ============================================================

DATASET_PATH = (
    r"C:\Users\somavarapu lokesh\PycharmProjects"
    r"\placement_prediction\dataset\final_preprocess_M2.csv"
)

IMAGE_FOLDER = (
    r"C:\Users\somavarapu lokesh\PycharmProjects"
    r"\placement_prediction\outputs"
    r"\Linear_Regression_CFNE_GD_Compare_M2"
)


# ============================================================
# 2. CREATE IMAGE OUTPUT FOLDER
# ============================================================

os.makedirs(IMAGE_FOLDER, exist_ok=True)

print("Image output folder:")
print(IMAGE_FOLDER)


# ============================================================
# 3. LOAD DATASET
# ============================================================

print("\nLoading dataset...")

data = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully.")
print("Original dataset shape:", data.shape)


# ============================================================
# 4. CHECK DATASET
# ============================================================

print("\nChecking dataset...")


# Count NaN values
total_nan = data.isna().sum().sum()

# Count infinite values
numeric_data = data.select_dtypes(include=[np.number])

total_inf = np.isinf(numeric_data.to_numpy()).sum()


print("Total NaN values:", total_nan)
print("Total infinite values:", total_inf)


# ============================================================
# 5. CONVERT DATA TO NUMERIC
# ============================================================

# Since Linear Regression requires numerical data,
# convert all columns to numeric where possible.

for column in data.columns:
    data[column] = pd.to_numeric(
        data[column],
        errors="coerce"
    )


# ============================================================
# 6. REPLACE INFINITE VALUES WITH NaN
# ============================================================

data = data.replace(
    [np.inf, -np.inf],
    np.nan
)


print("\nNaN values after numeric conversion:")

nan_counts = data.isna().sum()

print(
    nan_counts[nan_counts > 0]
)


# ============================================================
# 7. EXTRACT FEATURES AND TARGET
# ============================================================

# All columns except the last column = features
X_df = data.iloc[:, :-1].copy()

# Last column = target
y_series = data.iloc[:, -1].copy()


# ============================================================
# 8. HANDLE INVALID TARGET VALUES
# ============================================================

# Target values cannot be imputed safely for this model.
# Therefore, remove rows where target is missing.

valid_target = y_series.notna()

removed_target_rows = (~valid_target).sum()

if removed_target_rows > 0:

    print(
        "\nRemoving rows with missing target:",
        removed_target_rows
    )

    X_df = X_df.loc[valid_target].reset_index(drop=True)

    y_series = y_series.loc[valid_target].reset_index(drop=True)


# ============================================================
# 9. HANDLE MISSING FEATURE VALUES
# ============================================================

# Replace NaN values in feature columns with
# the median of that feature.

missing_features_before = X_df.isna().sum().sum()

print(
    "\nMissing feature values before imputation:",
    missing_features_before
)


if missing_features_before > 0:

    for column in X_df.columns:

        if X_df[column].isna().any():

            median_value = X_df[column].median()

            # If the whole column is NaN,
            # use 0 as a safe fallback.
            if pd.isna(median_value):
                median_value = 0.0

            X_df[column] = X_df[column].fillna(
                median_value
            )


missing_features_after = X_df.isna().sum().sum()


print(
    "Missing feature values after imputation:",
    missing_features_after
)


# ============================================================
# 10. FINAL INFINITE VALUE CHECK
# ============================================================

X_array_check = X_df.to_numpy(dtype=float)
y_array_check = y_series.to_numpy(dtype=float)


feature_inf = np.isinf(X_array_check).sum()
target_inf = np.isinf(y_array_check).sum()


print(
    "Infinite values in features:",
    feature_inf
)

print(
    "Infinite values in target:",
    target_inf
)


# ============================================================
# 11. REMOVE ANY REMAINING INVALID ROWS
# ============================================================

valid_rows = (
    np.isfinite(X_array_check).all(axis=1)
    &
    np.isfinite(y_array_check)
)


invalid_rows = (~valid_rows).sum()


if invalid_rows > 0:

    print(
        "\nRemoving remaining invalid rows:",
        invalid_rows
    )

    X_array_check = X_array_check[valid_rows]

    y_array_check = y_array_check[valid_rows]


# ============================================================
# 12. FINAL X AND y
# ============================================================

X = X_array_check

y = y_array_check


print("\nFinal dataset shape:")
print("X:", X.shape)
print("y:", y.shape)


# ============================================================
# 13. FINAL NaN CHECK
# ============================================================

print("\nFinal validation:")

print(
    "NaN in X:",
    np.isnan(X).sum()
)

print(
    "NaN in y:",
    np.isnan(y).sum()
)

print(
    "Infinite values in X:",
    np.isinf(X).sum()
)

print(
    "Infinite values in y:",
    np.isinf(y).sum()
)


# ============================================================
# 14. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("\nTrain-Test Split:")

print(
    "X_train:",
    X_train.shape
)

print(
    "X_test:",
    X_test.shape
)

print(
    "y_train:",
    y_train.shape
)

print(
    "y_test:",
    y_test.shape
)


# ============================================================
# 15. FEATURE SCALING
# ============================================================

scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(
    X_train
)


X_test_scaled = scaler.transform(
    X_test
)


# ============================================================
# 16. CLOSED-FORM NORMAL EQUATION
# ============================================================

print(
    "\nRunning Closed-Form Normal Equation..."
)


# Add bias/intercept column

X_train_bias = np.c_[
    np.ones((X_train_scaled.shape[0], 1)),
    X_train_scaled
]


X_test_bias = np.c_[
    np.ones((X_test_scaled.shape[0], 1)),
    X_test_scaled
]


# ------------------------------------------------------------
# Normal Equation
#
# theta = (X^T X)^(-1) X^T y
#
# np.linalg.pinv() is used instead of np.linalg.inv()
# because pseudo-inverse is more stable.
# ------------------------------------------------------------

theta = np.linalg.pinv(
    X_train_bias
).dot(
    y_train
)


# ============================================================
# 17. NORMAL EQUATION PREDICTION
# ============================================================

pred_normal = X_test_bias.dot(
    theta
)


# Check prediction values

print(
    "NaN in Normal Equation predictions:",
    np.isnan(pred_normal).sum()
)

print(
    "Infinite values in Normal Equation predictions:",
    np.isinf(pred_normal).sum()
)


# ============================================================
# 18. NORMAL EQUATION METRICS
# ============================================================

mse_normal = mean_squared_error(
    y_test,
    pred_normal
)


r2_normal = r2_score(
    y_test,
    pred_normal
)


print(
    "\n------ Closed Form Normal Equation ------"
)


print("\nCoefficients:")

print(theta)


print(
    "\nMSE:",
    mse_normal
)


print(
    "R2 Score:",
    r2_normal
)


# ============================================================
# 19. GRADIENT DESCENT
# ============================================================

print(
    "\nRunning Gradient Descent..."
)


X_train_gd = np.c_[
    np.ones((X_train_scaled.shape[0], 1)),
    X_train_scaled
]


X_test_gd = np.c_[
    np.ones((X_test_scaled.shape[0], 1)),
    X_test_scaled
]


# Number of training samples

m = len(y_train)


# Initialize theta

theta_gd = np.zeros(
    X_train_gd.shape[1]
)


# Learning rate

learning_rate = 0.01


# Number of epochs

epochs = 1000


# ============================================================
# 20. STORE LOSS
# ============================================================

loss_history = []


# ============================================================
# 21. GRADIENT DESCENT ITERATIONS
# ============================================================

for epoch in range(epochs):

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    predictions = X_train_gd.dot(
        theta_gd
    )


    # --------------------------------------------------------
    # Error
    # --------------------------------------------------------

    errors = (
        predictions - y_train
    )


    # --------------------------------------------------------
    # Gradient
    # --------------------------------------------------------

    gradients = (
        (2 / m)
        *
        X_train_gd.T.dot(errors)
    )


    # --------------------------------------------------------
    # Update parameters
    # --------------------------------------------------------

    theta_gd -= (
        learning_rate * gradients
    )


    # --------------------------------------------------------
    # Calculate MSE
    # --------------------------------------------------------

    current_predictions = X_train_gd.dot(
        theta_gd
    )


    current_errors = (
        current_predictions - y_train
    )


    loss = np.mean(
        current_errors ** 2
    )


    loss_history.append(
        loss
    )


# ============================================================
# 22. GRADIENT DESCENT PREDICTION
# ============================================================

pred_gd = X_test_gd.dot(
    theta_gd
)


# Check predictions

print(
    "NaN in Gradient Descent predictions:",
    np.isnan(pred_gd).sum()
)

print(
    "Infinite values in Gradient Descent predictions:",
    np.isinf(pred_gd).sum()
)


# ============================================================
# 23. GRADIENT DESCENT METRICS
# ============================================================

mse_gd = mean_squared_error(
    y_test,
    pred_gd
)


r2_gd = r2_score(
    y_test,
    pred_gd
)


print(
    "\n------ Gradient Descent ------"
)


print("\nCoefficients:")

print(theta_gd)


print(
    "\nMSE:",
    mse_gd
)


print(
    "R2 Score:",
    r2_gd
)


# ============================================================
# 24. COMPARISON
# ============================================================

print(
    "\n=========== Comparison ==========="
)


print(
    "\nNormal Equation"
)


print(
    "MSE =",
    mse_normal
)


print(
    "R2 =",
    r2_normal
)


print(
    "\nGradient Descent"
)


print(
    "MSE =",
    mse_gd
)


print(
    "R2 =",
    r2_gd
)


# ============================================================
# 25. IMAGE 1
# ACTUAL VS PREDICTED VALUES
# ============================================================

plt.figure(
    figsize=(8, 6)
)


plt.scatter(
    y_test,
    pred_normal,
    alpha=0.5,
    label="Normal Equation"
)


plt.scatter(
    y_test,
    pred_gd,
    alpha=0.5,
    label="Gradient Descent"
)


# Perfect prediction line

minimum = min(
    y_test.min(),
    pred_normal.min(),
    pred_gd.min()
)


maximum = max(
    y_test.max(),
    pred_normal.max(),
    pred_gd.max()
)


plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--",
    label="Perfect Prediction"
)


plt.xlabel(
    "Actual Values"
)


plt.ylabel(
    "Predicted Values"
)


plt.title(
    "Actual vs Predicted Values"
)


plt.legend()


plt.grid(True)


plt.tight_layout()


image1 = os.path.join(
    IMAGE_FOLDER,
    "actual_vs_predicted.png"
)


plt.savefig(
    image1,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    "\nImage saved:"
)


print(
    image1
)


# ============================================================
# 26. IMAGE 2
# RESIDUAL COMPARISON
# ============================================================

# Residual = Actual - Predicted

normal_residuals = (
    y_test - pred_normal
)


gd_residuals = (
    y_test - pred_gd
)


plt.figure(
    figsize=(9, 6)
)


plt.scatter(
    pred_normal,
    normal_residuals,
    alpha=0.5,
    label="Normal Equation"
)


plt.scatter(
    pred_gd,
    gd_residuals,
    alpha=0.5,
    label="Gradient Descent"
)


plt.axhline(
    y=0,
    linestyle="--"
)


plt.xlabel(
    "Predicted Values"
)


plt.ylabel(
    "Residuals"
)


plt.title(
    "Residual Comparison"
)


plt.legend()


plt.grid(True)


plt.tight_layout()


image2 = os.path.join(
    IMAGE_FOLDER,
    "residual_comparison.png"
)


plt.savefig(
    image2,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    "Image saved:"
)


print(
    image2
)


# ============================================================
# 27. IMAGE 3
# GRADIENT DESCENT LOSS CURVE
# ============================================================

plt.figure(
    figsize=(9, 6)
)


plt.plot(
    range(1, epochs + 1),
    loss_history
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Mean Squared Error"
)


plt.title(
    "Gradient Descent Convergence"
)


plt.grid(True)


plt.tight_layout()


image3 = os.path.join(
    IMAGE_FOLDER,
    "gradient_descent_loss.png"
)


plt.savefig(
    image3,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    "Image saved:"
)


print(
    image3
)


# ============================================================
# 28. SAVE IMAGE INFORMATION
# ============================================================

image_info = pd.DataFrame({

    "Image": [

        "actual_vs_predicted.png",

        "residual_comparison.png",

        "gradient_descent_loss.png"

    ],

    "Description": [

        "Actual values versus predictions from both methods",

        "Residual comparison between Normal Equation and Gradient Descent",

        "MSE loss across Gradient Descent epochs"

    ]

})


image_info_path = os.path.join(
    IMAGE_FOLDER,
    "image_information.csv"
)


image_info.to_csv(
    image_info_path,
    index=False
)


print(
    "\nImage information saved:"
)


print(
    image_info_path
)


# ============================================================
# 29. FINAL VALIDATION
# ============================================================

print(
    "\n=========================================="
)

print(
    "FINAL VALIDATION"
)

print(
    "=========================================="
)


print(
    "\nNaN in y_test:",
    np.isnan(y_test).sum()
)


print(
    "NaN in pred_normal:",
    np.isnan(pred_normal).sum()
)


print(
    "NaN in pred_gd:",
    np.isnan(pred_gd).sum()
)


print(
    "Infinite values in y_test:",
    np.isinf(y_test).sum()
)


print(
    "Infinite values in pred_normal:",
    np.isinf(pred_normal).sum()
)


print(
    "Infinite values in pred_gd:",
    np.isinf(pred_gd).sum()
)


# ============================================================
# 30. FINAL MESSAGE
# ============================================================

print(
    "\n=========================================="
)

print(
    "PROCESS COMPLETED SUCCESSFULLY"
)

print(
    "=========================================="
)


print(
    "\nAll images are stored in ONE folder:"
)


print(
    IMAGE_FOLDER
)


print(
    "\nGenerated images:"
)


print(
    "1. actual_vs_predicted.png"
)


print(
    "2. residual_comparison.png"
)


print(
    "3. gradient_descent_loss.png"
)


print(
    "4. image_information.csv"
)


print(
    "\nOriginal dataset was NOT modified."
)