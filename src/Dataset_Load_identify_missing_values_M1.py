import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\somavarapu lokesh\PycharmProjects\placement_prediction\uploads\placement_predict_50K_Raw (2) (1).csv")

print("--- First 5 Rows ---")
print(df.head())

print("---print 6 columns---")
subset =df.iloc[:,0:6]
print(subset)

missing_counts =df.isnull().sum()
print("----Missing Values per Column: ----")

print(missing_counts)
print("-" * 40)

duplicate_rows = df[df.duplicated()]
print(f"Total duplicate rows detected: (len(dupliacte_rows)")
print(duplicate_rows)
print("-" * 40)
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap="viridis")
plt.title("Missing Values Heatmap")
plt.show()
