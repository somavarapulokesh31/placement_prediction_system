# ============================================================
# PLACEMENT PREDICTION SYSTEM
# K-MEANS AND K-MEANS++
# ELBOW + SILHOUETTE + PCA + ACCURACY
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    accuracy_score
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. INPUT PATH
# ============================================================

INPUT_FILE = (
    r"C:\Users\somavarapu lokesh\PycharmProjects\placement_prediction"
    r"\dataset\final_preprocess_M2.csv"
)


# ============================================================
# 2. MAIN OUTPUT PATH
# ============================================================

OUTPUT_FOLDER = (
    r"C:\Users\somavarapu lokesh\PycharmProjects\placement_prediction"
    r"\outputs\K_Means_K++Means_Elbow_Silhoute_M4_Outputs"
)


# ============================================================
# 3. OUTPUT SUB-FOLDERS
# ============================================================

ACCURACY_FOLDER = os.path.join(
    OUTPUT_FOLDER, "Accuracy"
)

CHARTS_FOLDER = os.path.join(
    OUTPUT_FOLDER, "charts"
)

CLUSTER_RESULTS_FOLDER = os.path.join(
    OUTPUT_FOLDER, "cluster_results"
)

ELBOW_FOLDER = os.path.join(
    OUTPUT_FOLDER, "Elbow"
)

KMEANS_FOLDER = os.path.join(
    OUTPUT_FOLDER, "KMeans"
)

KMEANS_PP_FOLDER = os.path.join(
    OUTPUT_FOLDER, "KMeansPlusPlus"
)

METRICS_FOLDER = os.path.join(
    OUTPUT_FOLDER, "metrics"
)

PCA_FOLDER = os.path.join(
    OUTPUT_FOLDER, "pca"
)

SILHOUETTE_FOLDER = os.path.join(
    OUTPUT_FOLDER, "Silhouette"
)


# Create all folders automatically
ALL_FOLDERS = [
    OUTPUT_FOLDER,
    ACCURACY_FOLDER,
    CHARTS_FOLDER,
    CLUSTER_RESULTS_FOLDER,
    ELBOW_FOLDER,
    KMEANS_FOLDER,
    KMEANS_PP_FOLDER,
    METRICS_FOLDER,
    PCA_FOLDER,
    SILHOUETTE_FOLDER
]

for folder in ALL_FOLDERS:
    os.makedirs(folder, exist_ok=True)


# ============================================================
# 4. PATH CHECK
# ============================================================

print("\n" + "=" * 75)
print("PATH CHECK")
print("=" * 75)

print(
    "Input dataset exists :",
    os.path.isfile(INPUT_FILE)
)

print(
    "Output folder        :",
    OUTPUT_FOLDER
)

if not os.path.isfile(INPUT_FILE):

    raise FileNotFoundError(
        "\nDataset not found at:\n"
        + INPUT_FILE
    )


# ============================================================
# 5. LOAD DATASET
# ============================================================

print("\n" + "=" * 75)
print("K-MEANS AND K-MEANS++")
print("PLACEMENT PREDICTION - PREPROCESSED DATASET")
print("=" * 75)

data = pd.read_csv(
    INPUT_FILE
)

print(
    "\nPreprocessed dataset loaded successfully."
)

print(
    "Rows    :",
    data.shape[0]
)

print(
    "Columns :",
    data.shape[1]
)


# ============================================================
# 6. DISPLAY COLUMNS
# ============================================================

print("\nAvailable columns:")
print("-" * 75)

for i, column in enumerate(
    data.columns,
    1
):
    print(
        i,
        ".",
        column
    )


# ============================================================
# 7. NORMALIZE COLUMN NAMES
# ============================================================

def normalize_column_name(column):

    return (
        str(column)
        .strip()
        .lower()
        .replace("_", "")
        .replace(" ", "")
        .replace("-", "")
    )


normalized_columns = {
    normalize_column_name(column): column
    for column in data.columns
}


# ============================================================
# 8. REQUIRED FEATURES
# ============================================================

required_features = [
    "CGPA",
    "HistoryOfBacklogs",
    "Internships",
    "AptitudeTestScore"
]


# ============================================================
# 9. FIND REQUIRED FEATURES
# ============================================================

feature_columns = []

for feature in required_features:

    normalized_feature = (
        normalize_column_name(feature)
    )

    if normalized_feature in normalized_columns:

        feature_columns.append(
            normalized_columns[
                normalized_feature
            ]
        )

    else:

        raise ValueError(
            "\nRequired feature not found: "
            + feature
        )


# ============================================================
# 10. DISPLAY FEATURES
# ============================================================

print("\n" + "=" * 75)
print("FEATURES USED FOR CLUSTERING")
print("=" * 75)

for feature in feature_columns:

    print(
        "✓",
        feature
    )


# ============================================================
# 11. SELECT FEATURES
# ============================================================

X = data[
    feature_columns
].copy()


# ============================================================
# 12. CONVERT FEATURES TO NUMERIC
# ============================================================

for column in feature_columns:

    X[column] = pd.to_numeric(
        X[column],
        errors="coerce"
    )


# ============================================================
# 13. HANDLE INFINITE VALUES
# ============================================================

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)


# ============================================================
# 14. CHECK MISSING VALUES
# ============================================================

print("\nMissing values before filling:")

print(
    X.isnull().sum()
)


# ============================================================
# 15. FILL MISSING VALUES WITH MEDIAN
# ============================================================

for column in feature_columns:

    X[column] = X[column].fillna(
        X[column].median()
    )


print(
    "\nMissing values after filling:"
)

print(
    X.isnull().sum()
)


# ============================================================
# 16. STANDARDIZATION
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X
)

print(
    "\nFeature standardisation completed."
)


# ============================================================
# 17. K VALUES
# ============================================================

K_VALUES = range(
    2,
    11
)


# ============================================================
# 18. K-MEANS ANALYSIS
# ============================================================

print("\n" + "=" * 75)
print("K-MEANS ANALYSIS")
print("=" * 75)

kmeans_inertia = []

kmeans_silhouette = []

for k in K_VALUES:

    model = KMeans(
        n_clusters=k,
        init="random",
        n_init=10,
        random_state=42
    )

    labels = model.fit_predict(
        X_scaled
    )

    inertia = model.inertia_

    silhouette = silhouette_score(
        X_scaled,
        labels
    )

    kmeans_inertia.append(
        inertia
    )

    kmeans_silhouette.append(
        silhouette
    )

    print(
        f"K = {k:2d} | "
        f"Inertia = {inertia:.4f} | "
        f"Silhouette = {silhouette:.4f}"
    )


# ============================================================
# 19. K-MEANS++ ANALYSIS
# ============================================================

print("\n" + "=" * 75)
print("K-MEANS++ ANALYSIS")
print("=" * 75)

kmeans_pp_inertia = []

kmeans_pp_silhouette = []

for k in K_VALUES:

    model = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=10,
        random_state=42
    )

    labels = model.fit_predict(
        X_scaled
    )

    inertia = model.inertia_

    silhouette = silhouette_score(
        X_scaled,
        labels
    )

    kmeans_pp_inertia.append(
        inertia
    )

    kmeans_pp_silhouette.append(
        silhouette
    )

    print(
        f"K = {k:2d} | "
        f"Inertia = {inertia:.4f} | "
        f"Silhouette = {silhouette:.4f}"
    )


# ============================================================
# 20. BEST K
# ============================================================

k_values_list = list(
    K_VALUES
)

best_k_kmeans = k_values_list[
    int(
        np.argmax(
            kmeans_silhouette
        )
    )
]

best_k_kmeans_pp = k_values_list[
    int(
        np.argmax(
            kmeans_pp_silhouette
        )
    )
]

print("\n" + "=" * 75)
print("SELECTED NUMBER OF CLUSTERS")
print("=" * 75)

print(
    "K-Means best K   :",
    best_k_kmeans
)

print(
    "K-Means++ best K :",
    best_k_kmeans_pp
)


# ============================================================
# 21. ELBOW - K-MEANS
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    k_values_list,
    kmeans_inertia,
    marker="o"
)

plt.xlabel(
    "Number of Clusters (K)"
)

plt.ylabel(
    "Inertia"
)

plt.title(
    "Elbow Method - K-Means"
)

plt.xticks(
    k_values_list
)

plt.grid(
    True
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        ELBOW_FOLDER,
        "elbow_kmeans.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 22. ELBOW - K-MEANS++
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    k_values_list,
    kmeans_pp_inertia,
    marker="o"
)

plt.xlabel(
    "Number of Clusters (K)"
)

plt.ylabel(
    "Inertia"
)

plt.title(
    "Elbow Method - K-Means++"
)

plt.xticks(
    k_values_list
)

plt.grid(
    True
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        ELBOW_FOLDER,
        "elbow_kmeans_plus_plus.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 23. SILHOUETTE - K-MEANS
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    k_values_list,
    kmeans_silhouette,
    marker="o"
)

plt.xlabel(
    "Number of Clusters (K)"
)

plt.ylabel(
    "Silhouette Score"
)

plt.title(
    "Silhouette Score - K-Means"
)

plt.xticks(
    k_values_list
)

plt.grid(
    True
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        SILHOUETTE_FOLDER,
        "silhouette_kmeans.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 24. SILHOUETTE - K-MEANS++
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    k_values_list,
    kmeans_pp_silhouette,
    marker="o"
)

plt.xlabel(
    "Number of Clusters (K)"
)

plt.ylabel(
    "Silhouette Score"
)

plt.title(
    "Silhouette Score - K-Means++"
)

plt.xticks(
    k_values_list
)

plt.grid(
    True
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        SILHOUETTE_FOLDER,
        "silhouette_kmeans_plus_plus.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 25. FINAL K-MEANS MODEL
# ============================================================

kmeans_model = KMeans(
    n_clusters=best_k_kmeans,
    init="random",
    n_init=10,
    random_state=42
)

kmeans_labels = kmeans_model.fit_predict(
    X_scaled
)


# ============================================================
# 26. FINAL K-MEANS++ MODEL
# ============================================================

kmeans_pp_model = KMeans(
    n_clusters=best_k_kmeans_pp,
    init="k-means++",
    n_init=10,
    random_state=42
)

kmeans_pp_labels = kmeans_pp_model.fit_predict(
    X_scaled
)


# ============================================================
# 27. CLUSTER RESULT DATA
# ============================================================

kmeans_result = X.copy()

kmeans_result[
    "KMeans_Cluster"
] = kmeans_labels + 1


kmeans_pp_result = X.copy()

kmeans_pp_result[
    "KMeansPlusPlus_Cluster"
] = kmeans_pp_labels + 1


# ============================================================
# 28. SAVE CLUSTER RESULTS
# ============================================================

kmeans_result.to_csv(
    os.path.join(
        KMEANS_FOLDER,
        "kmeans_clustered_data.csv"
    ),
    index=False
)

kmeans_pp_result.to_csv(
    os.path.join(
        KMEANS_PP_FOLDER,
        "kmeans_plus_plus_clustered_data.csv"
    ),
    index=False
)

kmeans_result.to_csv(
    os.path.join(
        CLUSTER_RESULTS_FOLDER,
        "kmeans_clustered_data.csv"
    ),
    index=False
)

kmeans_pp_result.to_csv(
    os.path.join(
        CLUSTER_RESULTS_FOLDER,
        "kmeans_plus_plus_clustered_data.csv"
    ),
    index=False
)


# ============================================================
# 29. CLUSTER COUNTS
# ============================================================

kmeans_counts = (
    pd.Series(
        kmeans_labels + 1
    )
    .value_counts()
    .sort_index()
)

kmeans_pp_counts = (
    pd.Series(
        kmeans_pp_labels + 1
    )
    .value_counts()
    .sort_index()
)

cluster_numbers = list(
    range(
        1,
        max(
            best_k_kmeans,
            best_k_kmeans_pp
        ) + 1
    )
)

cluster_counts = pd.DataFrame({
    "Cluster": cluster_numbers
})

cluster_counts[
    "KMeans_Count"
] = (
    cluster_counts["Cluster"]
    .map(kmeans_counts)
    .fillna(0)
    .astype(int)
)

cluster_counts[
    "KMeansPlusPlus_Count"
] = (
    cluster_counts["Cluster"]
    .map(kmeans_pp_counts)
    .fillna(0)
    .astype(int)
)

cluster_counts.to_csv(
    os.path.join(
        CLUSTER_RESULTS_FOLDER,
        "cluster_counts.csv"
    ),
    index=False
)


# ============================================================
# 30. FINAL K-MEANS METRICS
# ============================================================

km_silhouette = silhouette_score(
    X_scaled,
    kmeans_labels
)

km_calinski = calinski_harabasz_score(
    X_scaled,
    kmeans_labels
)

km_davies = davies_bouldin_score(
    X_scaled,
    kmeans_labels
)


# ============================================================
# 31. FINAL K-MEANS++ METRICS
# ============================================================

pp_silhouette = silhouette_score(
    X_scaled,
    kmeans_pp_labels
)

pp_calinski = calinski_harabasz_score(
    X_scaled,
    kmeans_pp_labels
)

pp_davies = davies_bouldin_score(
    X_scaled,
    kmeans_pp_labels
)


# ============================================================
# 32. SAVE K-MEANS METRICS
# ============================================================

kmeans_metrics = pd.DataFrame({
    "Method": [
        "K-Means"
    ],
    "Number_of_Clusters": [
        best_k_kmeans
    ],
    "Inertia": [
        kmeans_model.inertia_
    ],
    "Silhouette_Score": [
        km_silhouette
    ],
    "Calinski_Harabasz_Score": [
        km_calinski
    ],
    "Davies_Bouldin_Score": [
        km_davies
    ]
})

kmeans_metrics.to_csv(
    os.path.join(
        KMEANS_FOLDER,
        "kmeans_metrics.csv"
    ),
    index=False
)


# ============================================================
# 33. SAVE K-MEANS++ METRICS
# ============================================================

kmeans_pp_metrics = pd.DataFrame({
    "Method": [
        "K-Means++"
    ],
    "Number_of_Clusters": [
        best_k_kmeans_pp
    ],
    "Inertia": [
        kmeans_pp_model.inertia_
    ],
    "Silhouette_Score": [
        pp_silhouette
    ],
    "Calinski_Harabasz_Score": [
        pp_calinski
    ],
    "Davies_Bouldin_Score": [
        pp_davies
    ]
})

kmeans_pp_metrics.to_csv(
    os.path.join(
        KMEANS_PP_FOLDER,
        "kmeans_plus_plus_metrics.csv"
    ),
    index=False
)


# ============================================================
# 34. COMBINED METRICS
# ============================================================

metrics_comparison = pd.DataFrame({
    "Method": [
        "K-Means",
        "K-Means++"
    ],
    "Best_K": [
        best_k_kmeans,
        best_k_kmeans_pp
    ],
    "Inertia": [
        kmeans_model.inertia_,
        kmeans_pp_model.inertia_
    ],
    "Silhouette_Score": [
        km_silhouette,
        pp_silhouette
    ],
    "Calinski_Harabasz_Score": [
        km_calinski,
        pp_calinski
    ],
    "Davies_Bouldin_Score": [
        km_davies,
        pp_davies
    ]
})

metrics_comparison.to_csv(
    os.path.join(
        METRICS_FOLDER,
        "clustering_metrics_comparison.csv"
    ),
    index=False
)


# ============================================================
# 35. 3-D K-MEANS GRAPH
# ============================================================

fig = plt.figure(
    figsize=(11, 8)
)

ax = fig.add_subplot(
    111,
    projection="3d"
)

scatter = ax.scatter(
    X[feature_columns[0]],
    X[feature_columns[1]],
    X[feature_columns[2]],
    c=kmeans_labels,
    s=20,
    alpha=0.7
)

ax.set_xlabel(
    feature_columns[0]
)

ax.set_ylabel(
    feature_columns[1]
)

ax.set_zlabel(
    feature_columns[2]
)

ax.set_title(
    "K-Means Clustering"
)

fig.colorbar(
    scatter,
    ax=ax,
    label="Cluster"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        CHARTS_FOLDER,
        "kmeans_3D_clustering.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 36. 3-D K-MEANS++ GRAPH
# ============================================================

fig = plt.figure(
    figsize=(11, 8)
)

ax = fig.add_subplot(
    111,
    projection="3d"
)

scatter = ax.scatter(
    X[feature_columns[0]],
    X[feature_columns[1]],
    X[feature_columns[2]],
    c=kmeans_pp_labels,
    s=20,
    alpha=0.7
)

ax.set_xlabel(
    feature_columns[0]
)

ax.set_ylabel(
    feature_columns[1]
)

ax.set_zlabel(
    feature_columns[2]
)

ax.set_title(
    "K-Means++ Clustering"
)

fig.colorbar(
    scatter,
    ax=ax,
    label="Cluster"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        CHARTS_FOLDER,
        "kmeans_plus_plus_3D_clustering.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 37. PCA
# ============================================================

pca = PCA(
    n_components=2
)

X_pca = pca.fit_transform(
    X_scaled
)


# ============================================================
# 38. K-MEANS PCA GRAPH
# ============================================================

plt.figure(
    figsize=(10, 7)
)

scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=kmeans_labels,
    s=20,
    alpha=0.7
)

plt.xlabel(
    "Principal Component 1"
)

plt.ylabel(
    "Principal Component 2"
)

plt.title(
    "K-Means Clusters - PCA Visualization"
)

plt.colorbar(
    scatter,
    label="Cluster"
)

plt.grid(
    True
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PCA_FOLDER,
        "kmeans_PCA_clustering.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 39. K-MEANS++ PCA GRAPH
# ============================================================

plt.figure(
    figsize=(10, 7)
)

scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=kmeans_pp_labels,
    s=20,
    alpha=0.7
)

plt.xlabel(
    "Principal Component 1"
)

plt.ylabel(
    "Principal Component 2"
)

plt.title(
    "K-Means++ Clusters - PCA Visualization"
)

plt.colorbar(
    scatter,
    label="Cluster"
)

plt.grid(
    True
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PCA_FOLDER,
        "kmeans_plus_plus_PCA_clustering.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 40. FIND PLACEMENT TARGET
# ============================================================

target_candidates = [
    "PlacementStatus",
    "Placement",
    "Placed",
    "Status",
    "Target"
]

target_column = None

for candidate in target_candidates:

    normalized_candidate = (
        normalize_column_name(
            candidate
        )
    )

    if normalized_candidate in normalized_columns:

        target_column = normalized_columns[
            normalized_candidate
        ]

        break


# ============================================================
# 41. ROBUST CLUSTER MATCHING ACCURACY
# ============================================================

def cluster_matching_accuracy(
    true_values,
    cluster_values
):

    true_values = pd.Series(
        true_values
    ).reset_index(drop=True)

    cluster_values = pd.Series(
        cluster_values
    ).reset_index(drop=True)

    if len(true_values) != len(
        cluster_values
    ):

        raise ValueError(
            "Target values and cluster "
            "values have different lengths."
        )

    # Convert both values to strings so sklearn
    # receives one consistent target type.
    true_values = (
        true_values
        .astype("string")
        .fillna("MISSING")
        .str.strip()
    )

    predicted_values = pd.Series(
        index=range(
            len(cluster_values)
        ),
        dtype="string"
    )

    for cluster in sorted(
        cluster_values.dropna().unique()
    ):

        indexes = np.where(
            cluster_values.to_numpy()
            == cluster
        )[0]

        cluster_targets = (
            true_values.iloc[
                indexes
            ]
        )

        majority = cluster_targets.mode(
            dropna=False
        )

        if not majority.empty:

            predicted_values.iloc[
                indexes
            ] = str(
                majority.iloc[0]
            )

    predicted_values = (
        predicted_values
        .fillna("UNKNOWN")
        .astype("string")
    )

    true_array = (
        true_values.to_numpy(
            dtype=str
        )
    )

    predicted_array = (
        predicted_values.to_numpy(
            dtype=str
        )
    )

    return accuracy_score(
        true_array,
        predicted_array
    )


# ============================================================
# 42. CALCULATE ACCURACY
# ============================================================

print("\n" + "=" * 75)
print("PLACEMENT CLUSTER MATCHING ACCURACY")
print("=" * 75)

if target_column is not None:

    print(
        "Target column:",
        target_column
    )

    target = data[
        target_column
    ].copy()

    valid = target.notna()

    target_valid = (
        target[
            valid
        ]
        .reset_index(drop=True)
    )

    km_labels_valid = (
        pd.Series(
            kmeans_labels,
            index=data.index
        )[valid]
        .reset_index(drop=True)
    )

    pp_labels_valid = (
        pd.Series(
            kmeans_pp_labels,
            index=data.index
        )[valid]
        .reset_index(drop=True)
    )

    print(
        "Target data type:",
        target_valid.dtype
    )

    print(
        "Target unique values:",
        target_valid.unique()
    )

    km_accuracy = (
        cluster_matching_accuracy(
            target_valid,
            km_labels_valid
        )
    )

    pp_accuracy = (
        cluster_matching_accuracy(
            target_valid,
            pp_labels_valid
        )
    )

    print(
        f"K-Means   : "
        f"{km_accuracy * 100:.2f}%"
    )

    print(
        f"K-Means++ : "
        f"{pp_accuracy * 100:.2f}%"
    )

    accuracy_result = pd.DataFrame({

        "Method": [
            "K-Means",
            "K-Means++"
        ],

        "Cluster_Matching_Accuracy": [
            km_accuracy,
            pp_accuracy
        ],

        "Accuracy_Percentage": [
            km_accuracy * 100,
            pp_accuracy * 100
        ]

    })

    accuracy_result.to_csv(
        os.path.join(
            ACCURACY_FOLDER,
            "placement_cluster_matching_accuracy.csv"
        ),
        index=False
    )

else:

    print(
        "\nNo PlacementStatus/Placement/"
        "Placed/Status/Target column was found."
    )

    print(
        "Accuracy was not calculated."
    )


# ============================================================
# 43. CLUSTER COUNTS
# ============================================================

print("\n" + "=" * 75)
print("K-MEANS CLUSTER COUNTS")
print("=" * 75)

print(
    kmeans_result[
        "KMeans_Cluster"
    ]
    .value_counts()
    .sort_index()
)


print("\n" + "=" * 75)
print("K-MEANS++ CLUSTER COUNTS")
print("=" * 75)

print(
    kmeans_pp_result[
        "KMeansPlusPlus_Cluster"
    ]
    .value_counts()
    .sort_index()
)


# ============================================================
# 44. FINAL RESULTS
# ============================================================

print("\n" + "=" * 75)
print("FINAL CLUSTERING RESULTS")
print("=" * 75)


print("\nK-MEANS")
print("-" * 40)

print(
    "Best K:",
    best_k_kmeans
)

print(
    "Inertia:",
    round(
        kmeans_model.inertia_,
        4
    )
)

print(
    "Silhouette:",
    round(
        km_silhouette,
        4
    )
)

print(
    "Calinski-Harabasz:",
    round(
        km_calinski,
        4
    )
)

print(
    "Davies-Bouldin:",
    round(
        km_davies,
        4
    )
)


print("\nK-MEANS++")
print("-" * 40)

print(
    "Best K:",
    best_k_kmeans_pp
)

print(
    "Inertia:",
    round(
        kmeans_pp_model.inertia_,
        4
    )
)

print(
    "Silhouette:",
    round(
        pp_silhouette,
        4
    )
)

print(
    "Calinski-Harabasz:",
    round(
        pp_calinski,
        4
    )
)

print(
    "Davies-Bouldin:",
    round(
        pp_davies,
        4
    )
)


# ============================================================
# 45. OUTPUT LOCATIONS
# ============================================================

print("\n" + "=" * 75)
print("OUTPUTS SAVED")
print("=" * 75)

print(
    "\nMain folder:"
)

print(
    OUTPUT_FOLDER
)

print(
    "\nAccuracy:"
)

print(
    ACCURACY_FOLDER
)

print(
    "\nCharts:"
)

print(
    CHARTS_FOLDER
)

print(
    "\nCluster Results:"
)

print(
    CLUSTER_RESULTS_FOLDER
)

print(
    "\nElbow:"
)

print(
    ELBOW_FOLDER
)

print(
    "\nK-Means:"
)

print(
    KMEANS_FOLDER
)

print(
    "\nK-Means++:"
)

print(
    KMEANS_PP_FOLDER
)

print(
    "\nMetrics:"
)

print(
    METRICS_FOLDER
)

print(
    "\nPCA:"
)

print(
    PCA_FOLDER
)

print(
    "\nSilhouette:"
)

print(
    SILHOUETTE_FOLDER
)


# ============================================================
# 46. FINAL SUCCESS MESSAGE
# ============================================================

print(
    "\nOriginal/preprocessed input dataset "
    "was NOT modified."
)

print(
    "\nPROGRAM COMPLETED SUCCESSFULLY."
)

print(
    "\nAll requested K-Means outputs were generated."
)

# ============================================================
# END
# ============================================================
