import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
from sklearn.metrics import silhouette_samples, silhouette_score
from Functions import extract_data_from_mpa, SilhouetteScore_to_Confidence
from sklearn.neighbors import LocalOutlierFactor
from matplotlib.legend_handler import HandlerPathCollection

# Set file path and subject
#filepath = r'C:\Users\benja\Desktop\Speciale\Data\Første måling af Be10\2025_01_16_Benedikt\2025_01_16_Benedikt'
filepath = r'C:\Users\benja\Desktop\Speciale\Master-Thesis\Data\Første måling af Be10\2025_01_16_Benedikt\2025_01_16_Benedikt'
subject = "[CDAT0"

billeder_path = r'C:\Users\benja\Desktop\Speciale\Master-Thesis\Billeder'

# Load and filter raw files
files = os.listdir(filepath)
raw_files = [file for file in files if file.endswith(".txt.mpa")]

# Extract data from MPA files
data = extract_data_from_mpa(filepath, subject, file_index=2, info=None )

# KMeans clustering parameters
numberof_clusters = 4  # Optimal number of clusters

# Perform KMeans clustering
X = data[['E_final', 'dE']]
kmeans = KMeans(n_clusters=numberof_clusters, random_state=30, init='k-means++')
data['Cluster_KMeans'] = kmeans.fit_predict(X)
print(data[['E_final', 'dE', 'Cluster_KMeans']].head())

# Calculate Silhouette scores for each point
data['silhouette_score'] = silhouette_samples(X, data['Cluster_KMeans'])

# Dynamically identify the ROI cluster (bottom-right cluster)
centroids = kmeans.cluster_centers_
bottom_clusters = [cluster_id for cluster_id, centroid in enumerate(centroids) if centroid[1] < np.median(centroids[:, 1])]
roi_cluster_id = bottom_clusters[np.argmax([centroids[cluster_id][0] for cluster_id in bottom_clusters])]
print(f"Region of Interest (ROI) Cluster ID: {roi_cluster_id}")

# Focus on the ROI cluster and analyze its silhouette score
roi_cluster_data = data[data['Cluster_KMeans'] == roi_cluster_id]
roi_cluster_silhouette_score = roi_cluster_data['silhouette_score'].mean()
print(f"Average Silhouette Score for ROI Cluster: {roi_cluster_silhouette_score}")

# Calculate distance to centroid for the ROI cluster
roi_cluster_centroid = kmeans.cluster_centers_[roi_cluster_id]
roi_cluster_data['distance_to_centroid'] = np.linalg.norm(
    roi_cluster_data[['E_final', 'dE']].values - roi_cluster_centroid, axis=1
)
mean_distance_to_centroid = roi_cluster_data['distance_to_centroid'].mean()
print(f"Mean Distance to Centroid for ROI Cluster: {mean_distance_to_centroid}")

# Distance to other centroids
distances_to_other_centroids = np.linalg.norm(kmeans.cluster_centers_ - roi_cluster_centroid, axis=1)
print(f"Distance of ROI Cluster's centroid to other centroids: {distances_to_other_centroids}")

# ROI Cluster size
roi_cluster_size = roi_cluster_data.shape[0]
print(f"Number of points in ROI Cluster: {roi_cluster_size}")

# Filter points within a given radius (35 units)
roi_cluster_radius = 35
circle = plt.Circle((roi_cluster_centroid[0], roi_cluster_centroid[1]), roi_cluster_radius, color='blue', fill=False, linestyle='--', linewidth=2)

# Calculate Euclidean distance from each point to the centroid
distances = np.linalg.norm(roi_cluster_data[['E_final', 'dE']].values - roi_cluster_centroid, axis=1)
filtered_roi_cluster_data = roi_cluster_data[distances <= roi_cluster_radius]

# Plot the data without the marginal plots
fig, ax = plt.subplots(figsize=(8, 8))

# Main scatter plot
sc = ax.scatter(data["E_final"], data["dE"], c=data["counts"], cmap='viridis', s=2)
ax.set_xlabel(r"$E_{final} [keV]$")
ax.set_ylabel(r"$\Delta E [keV]$")
#title for number of k and number of cluster
ax.set_title(f"KMeans Clustering with {numberof_clusters} Clusters (k_eff = {numberof_clusters})")
ax.grid(True, linestyle='--', alpha=0.6)
ax.tick_params(direction="in", length=6, which="major")  # Major ticks longer
ax.tick_params(direction="in", length=3, which="minor")  # Minor ticks shorter
ax.xaxis.set_ticks_position("both")
ax.yaxis.set_ticks_position("both")
ax.minorticks_on()

# Add shaded areas for each cluster (alpha=0.3)
for cluster_id in data['Cluster_KMeans'].unique():
    cluster_points = data[data['Cluster_KMeans'] == cluster_id]
    points = cluster_points[['E_final', 'dE']].values
    if len(points) >= 3:
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        # Plot the shaded area for each cluster
        ax.fill(hull_points[:, 0], hull_points[:, 1], color=sns.color_palette("viridis", n_colors=numberof_clusters)[cluster_id], alpha=0.15, label=f'Cluster {cluster_id}')

# Add legend
plt.legend(loc='upper left')
# Colorbar
cbar = plt.colorbar(sc, ax=ax, orientation="vertical", fraction=0.05)
cbar.set_label("Counts")

# Final layout
plt.tight_layout()

plt.savefig(f'{billeder_path}\\KNNScatterPlotGrouping.pdf')
plt.show()

# Re-cluster the filtered points from the ROI cluster
filtered_X = filtered_roi_cluster_data[['E_final', 'dE']]
kmeans_filtered = KMeans(n_clusters=2, random_state=30, init='k-means++')
filtered_roi_cluster_data['Filtered_Cluster'] = kmeans_filtered.fit_predict(filtered_X)

# Silhouette score for re-clustered points
filtered_silhouette_score = silhouette_score(filtered_X, filtered_roi_cluster_data['Filtered_Cluster'])
print(f"Silhouette Score for the Filtered ROI Cluster (after re-clustering): {filtered_silhouette_score}")

# Calculate distance and variance for the original centroid (before filtering)
original_centroid = kmeans.cluster_centers_[roi_cluster_id]
distances_original = np.linalg.norm(roi_cluster_data[['E_final', 'dE']].values - original_centroid, axis=1)

# Mean distance and variance of the original cluster
mean_distance_original = np.mean(distances_original)
variance_original = np.var(distances_original)

# Standard errors for original centroid metrics
se_mean_distance_original = np.std(distances_original) / np.sqrt(len(distances_original))
se_variance_original = np.sqrt((2 * variance_original ** 2) / (len(distances_original) - 1))

print(f"Mean Distance to Centroid (Original): {mean_distance_original:.4f} ± {se_mean_distance_original:.4f}")
print(f"Variance of Distances (Original): {variance_original:.4f} ± {se_variance_original:.4f}")

# Calculate distance and variance for the filtered centroid (after filtering)
filtered_centroid = filtered_roi_cluster_data[['E_final', 'dE']].mean().values
distances_filtered = np.linalg.norm(filtered_roi_cluster_data[['E_final', 'dE']].values - filtered_centroid, axis=1)

# Mean distance and variance of the filtered cluster
mean_distance_filtered = np.mean(distances_filtered)
variance_filtered = np.var(distances_filtered)

# Standard errors for filtered centroid metrics
se_mean_distance_filtered = np.std(distances_filtered) / np.sqrt(len(distances_filtered))
se_variance_filtered = np.sqrt((2 * variance_filtered ** 2) / (len(distances_filtered) - 1))

print(f"Mean Distance to Centroid (Filtered): {mean_distance_filtered:.4f} ± {se_mean_distance_filtered:.4f}")
print(f"Variance of Distances (Filtered): {variance_filtered:.4f} ± {se_variance_filtered:.4f}")

# Count the filtered data points
N_filtered = filtered_roi_cluster_data["counts"].sum()
uncertainty_filtered = np.sqrt(N_filtered)
print(f"The sum of the filtered_roi_cluster_points for Be10 is: {N_filtered} ± {uncertainty_filtered}")

# Output the confidence of the silhouette score
print(f"Confidence of the Silhouette Score: {SilhouetteScore_to_Confidence(filtered_silhouette_score)}")

# Extract features (E_final and dE) for the ROI cluster
X_roi_cluster = roi_cluster_data[['E_final', 'dE']].values

# Apply Local Outlier Factor (LOF)
clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
y_pred = clf.fit_predict(X_roi_cluster)

# Calculate the negative outlier factor scores
X_scores = clf.negative_outlier_factor_

# Visualize the results
def update_legend_marker_size(handle, orig):
    "Customize size of the legend marker"
    handle.update_from(orig)
    handle.set_sizes([20])

plt.scatter(X_roi_cluster[:, 0], X_roi_cluster[:, 1], color="k", s=3.0, label="Data points")

# Plot circles with radius proportional to the outlier scores
radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
scatter = plt.scatter(
    X_roi_cluster[:, 0],
    X_roi_cluster[:, 1],
    s=1000 * radius,
    edgecolors="r",
    facecolors="none",
    label="Outlier scores",
)

plt.grid(True, linestyle='--', alpha=0.6)  # Dashed grid with transparency
plt.tick_params(direction="in", length=6, which="major")  # Major ticks longer
plt.tick_params(direction="in", length=3, which="minor")  # Minor ticks shorter

plt.minorticks_on()

plt.xlim([75, 195])
plt.ylim([25, 290])
plt.axis("tight")
plt.xlabel(r"$E_{final} [keV]$")
plt.ylabel(r"$\Delta E [keV]$")
plt.title(f"Local Outlier Factor (LOF) for ROI Cluster ")
plt.legend(
    handler_map={scatter: HandlerPathCollection(update_func=update_legend_marker_size)}
)
plt.savefig(f'{billeder_path}\\LOF10Beplot.pdf')

plt.show()

# Print outliers information
outliers = X_roi_cluster[y_pred == -1]
print(f"Outliers detected: {len(outliers)} points")
print(f"Outliers indices: {np.where(y_pred == -1)}")


fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Plot the full ROI cluster
ax1 = axes[0]
scatter1 = ax1.scatter(roi_cluster_data['E_final'], roi_cluster_data['dE'], c=roi_cluster_data['counts'], cmap='viridis', s=20)
ax1.set_title(f"ROI Cluster - Silhouette Score: {roi_cluster_silhouette_score:.3f}")
ax1.set_xlabel(r"$E_{final} [keV]$")
ax1.set_ylabel(r"$\Delta E [keV]$")
ax1.grid(True)


# Check if a legend exists
legend_exists = ax1.get_legend() is not None

for outlier in outliers:
    # Check if outlier is NOT in the filtered dataset
    if not np.any(np.all(outlier == filtered_roi_cluster_data[['E_final', 'dE']].values, axis=1)):
        # Find index of this outlier in X_roi_cluster to get its LOF score
        outlier_idx = np.where((X_roi_cluster == outlier).all(axis=1))[0][0]
        
        # Compute circle size based on LOF score
        outlier_radius = (X_scores.max() - X_scores[outlier_idx]) / (X_scores.max() - X_scores.min())

        # Determine if we need to add the legend label
        label = "Outlier scores" if not legend_exists else ""

        # Plot only this outlier
        ax1.scatter(
            outlier[0], outlier[1],
            s=1000 * outlier_radius,  # Scale size based on LOF score
            edgecolors="r",
            facecolors="none",
            label=label
        )



# Plot the filtered ROI cluster (without outliers)
ax2 = axes[1]
scatter2 = ax2.scatter(filtered_roi_cluster_data['E_final'], filtered_roi_cluster_data['dE'], c=filtered_roi_cluster_data['counts'], cmap='viridis', s=20)
ax2.set_title(f"Filtered ROI Cluster - Silhouette Score: {filtered_silhouette_score:.3f}")
ax2.set_xlabel(r"$E_{final} [keV]$")
ax2.set_ylabel(r"$\Delta E [keV]$")
ax2.grid(True)
print(f"Filtered ROI Cluster - Silhouette Score: {filtered_silhouette_score:.3f}")
# Add colorbars
cbar1 = plt.colorbar(scatter1, ax=ax1)
cbar1.set_label("Counts")
cbar2 = plt.colorbar(scatter2, ax=ax2)
cbar2.set_label("Counts")

#extra for the visuals
ax1.set_xlim([75, 195])
ax2.set_xlim([75, 195])
ax1.set_ylim([25, 290])
ax2.set_ylim([25, 290])

ax1.grid(True, linestyle='--', alpha=0.6)
ax1.tick_params(direction="in", length=6, which="major")  # Major ticks longer
ax1.tick_params(direction="in", length=3, which="minor")  # Minor ticks shorter
ax1.xaxis.set_ticks_position("both")
ax1.yaxis.set_ticks_position("both")
ax1.minorticks_on()

ax2.grid(True, linestyle='--', alpha=0.6)
ax2.tick_params(direction="in", length=6, which="major")  # Major ticks longer
ax2.tick_params(direction="in", length=3, which="minor")  # Minor ticks shorter
ax2.xaxis.set_ticks_position("both")
ax2.yaxis.set_ticks_position("both")
ax2.minorticks_on()



# Adjust layout
plt.tight_layout()
plt.savefig(f'{billeder_path}\\FilteringComparisonOf10BeGroups.pdf')
plt.show()

N_filtered = filtered_roi_cluster_data["counts"].sum()/10

# Calculate the uncertainty as the square root of the sum of counts
uncertainty_filtered = np.sqrt(N_filtered)

# Print the counts and uncertainty
print(f"The sum of the filtered_roi_cluster_points for Be10 is: {N_filtered} ± {uncertainty_filtered}")
print(F"Confidence of the Silhouette Score of ROI: {SilhouetteScore_to_Confidence(filtered_silhouette_score)}")

