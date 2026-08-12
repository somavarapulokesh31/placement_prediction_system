import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
import matplotlib.pyplot as plt

# Load Dataset
file_path = r"C:\Users\somavarapu lokesh\PycharmProjects\placement_prediction\uploads\placement_predict_50K_Raw (2) (1).csv"
df = pd.read_csv(file_path)

print("Original Dataset")
print("------------------------")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nData Types:")
print("------------------------")
print(df.dtypes)

print("\nMissing Values:")
print("------------------------")
print(df.isnull().sum())

print("\nDuplicate Records:", df.duplicated().sum())

# Remove Duplicates
df = df.drop_duplicates()

# Handle Missing Values in Numerical Columns
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns

for column in numerical_columns:
    df[column] = df[column].fillna(df[column].mean())

# Handle Missing Values in Categorical Columns
categorical_columns = df.select_dtypes(include=['object']).columns

for column in categorical_columns:
    df[column] = df[column].fillna(df[column].mode()[0])

# Remove Extra Spaces from Text Columns
for column in categorical_columns:
    df[column] = df[column].str.strip()

# Get Numeric Columns
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

print("\nNumeric Columns:")
print(list(numeric_columns))

# Standardization
standard_scaler = StandardScaler()
standardized = standard_scaler.fit_transform(df[numeric_columns])

for i, col in enumerate(numeric_columns):
    df[col + "_Standardized"] = standardized[:, i]

# Min-Max Scaling
minmax_scaler = MinMaxScaler()
scaled = minmax_scaler.fit_transform(df[numeric_columns])

for i, col in enumerate(numeric_columns):
    df[col + "_Scaled"] = scaled[:, i]

# Normalization
normalizer = Normalizer(norm='l2')
normalized = normalizer.fit_transform(df[numeric_columns])

for i, col in enumerate(numeric_columns):
    df[col + "_Normalized"] = normalized[:, i]

# Display Results
print("\nDisplay Results after Preprocessed Dataset")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nDataset Information")
df.info()

print("\nColumns in Dataset:")
print(df.columns)

print("\nMissing Values After Preprocessing")
print(df.isnull().sum())

print("\nDuplicate Records After Preprocessing")
print(df.duplicated().sum())

# Save Cleaned Dataset
output_file = r"C:\Users\somavarapu lokesh\PycharmProjects\placement_prediction\dataset\clean_minmax_stand_norma_M2.csv"
df.to_csv(output_file, index=False)

print("\nPreprocessed dataset saved successfully.")
print("Saved at:", output_file)

# Histogram
df.hist(figsize=(15, 12), bins=10, edgecolor='black')

plt.suptitle("Histogram of Preprocessed Placement Dataset")
plt.tight_layout()
plt.show()