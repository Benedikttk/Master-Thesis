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


import matplotlib as mpl

# Example: match LaTeX document font size of 12pt
mpl.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.titlesize': 16
})



# Set file path and subject
#filepath = r'C:\Users\benja\Desktop\Speciale\Data\Første måling af Be10\2025_01_16_Benedikt\2025_01_16_Benedikt'
#filepath = r'C:\Users\benja\Desktop\Speciale\Master-Thesis\Data\Første måling af Be10\2025_01_16_Benedikt\2025_01_16_Benedikt'
filepath = r'C:\Users\benja\Desktop\Speciale\NyBeeffdata\foralg'
subject = "[CDAT0"

billeder_path = r'C:\Users\benja\Desktop\Speciale\Billeder'

# Load and filter raw files
files = os.listdir(filepath)
raw_files = [file for file in files if file.endswith(".txt.mpa")]
print(raw_files)
# Extract data from MPA files
data = extract_data_from_mpa(filepath, subject, file_index=1, info=None )

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
fig, ax = plt.subplots(figsize=(12, 8))

ax.set_xlim(0,580)
ax.set_ylim(0,600)
# Main scatter plot
sc = ax.scatter(data["E_final"], data["dE"], c=data["counts"], cmap='viridis', s=2)
ax.set_xlabel(r"$E_{final} [Channel]$")
ax.set_ylabel(r"$\Delta E [Channel]$")
#title for number of k and number of cluster
ax.set_title(f"KMeans Clustering with {numberof_clusters} Clusters" r"($ k_{eff}$ =" f"{numberof_clusters})")
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


clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
y_pred = clf.fit_predict(X_roi_cluster)  # Apply LOF to the ROI cluster data

# Get the LOF scores (negative_outlier_factor_)
lof_scores = clf.negative_outlier_factor_

# The threshold is the value at which 10% of points are considered outliers
lof_threshold = np.percentile(lof_scores, 100 * 0.05)

# Print the cutoff value for the outlier score
print(f"Outlier score cutoff value: {lof_threshold:.3f}")



fig, ax = plt.subplots(figsize=(12, 8))

# Plot the data points (without LOF circles)
ax.scatter(X_roi_cluster[:, 0], X_roi_cluster[:, 1], color="k", s=3.0, label="Data points")

# Disable autoscale and set axis limits
ax.set_autoscale_on(True)


# Plot circles with radius proportional to the outlier scores
radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
scatter = ax.scatter(
    X_roi_cluster[:, 0],
    X_roi_cluster[:, 1],
    s=1000 * radius,  # Size of the points based on LOF score
    edgecolors="r",
    facecolors="none",
    label="Outlier scores",
)

# Add grid with dashed lines and adjust tick parameters
ax.grid(True, linestyle='--', alpha=0.6)
ax.tick_params(direction="in", length=6, which="major")  # Major ticks longer
ax.tick_params(direction="in", length=3, which="minor")  # Minor ticks shorter
ax.minorticks_on()

# Set axis labels and title
ax.set_xlabel(r"$E_{final} [Channel]$")
ax.set_ylabel(r"$\Delta E [Channel]$")
ax.set_title(f"Local Outlier Factor (LOF) for ROI Cluster")

# Add a legend
ax.legend(
    handler_map={scatter: HandlerPathCollection(update_func=update_legend_marker_size)}
)

# Save the plot to a file
plt.tight_layout()
plt.savefig(f'{billeder_path}\\LOF10Beplot.pdf')

# Show the plot
plt.show()




# Print outliers information
outliers = X_roi_cluster[y_pred == -1]
print(f"Outliers detected: {len(outliers)} points")
print(f"Outliers indices: {np.where(y_pred == -1)}")


fig, axes = plt.subplots(2,1, figsize=(12, 8), sharex=True)

# Plot the full ROI cluster
ax1 = axes[0]
scatter1 = ax1.scatter(roi_cluster_data['E_final'], roi_cluster_data['dE'], c=roi_cluster_data['counts'], cmap='viridis', s=20)
ax1.set_title(f"ROI Cluster - Silhouette Score: {roi_cluster_silhouette_score:.3f}")
ax1.set_xlabel(r"$E_{final} [Channel]$")
ax1.set_ylabel(r"$\Delta E [Channel]$")
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
ax2.set_xlabel(r"$E_{final} [Channel]$")
ax2.set_ylabel(r"$\Delta E [Channel]$")
ax2.grid(True)
print(f"Filtered ROI Cluster - Silhouette Score: {filtered_silhouette_score:.3f}")
# Add colorbars
cbar1 = plt.colorbar(scatter1, ax=ax1)
cbar1.set_label("Counts")
cbar2 = plt.colorbar(scatter2, ax=ax2)
cbar2.set_label("Counts")

#extra for the visuals
ax1.set_xlim([50, 200])
ax2.set_xlim([50, 200])
ax1.set_ylim([0, 300])
ax2.set_ylim([0, 300])

ax1.grid(True, linestyle='--', alpha=0.6)
ax1.tick_params(direction="in", length=6, which="major")  # Major ticks longer
ax1.tick_params(direction="in", length=3, which="minor")  # Minor ticks shorter
ax1.xaxis.set_ticks_position("both")
ax1.yaxis.set_ticks_position("both")
ax1.minorticks_on()
ax1.set_autoscale_on(True)

ax2.grid(True, linestyle='--', alpha=0.6)
ax2.tick_params(direction="in", length=6, which="major")  # Major ticks longer
ax2.tick_params(direction="in", length=3, which="minor")  # Minor ticks shorter
ax2.xaxis.set_ticks_position("both")
ax2.yaxis.set_ticks_position("both")
ax2.minorticks_on()
ax2.set_autoscale_on(True)



# Adjust layout
plt.tight_layout()
plt.savefig(f'{billeder_path}\\FilteringComparisonOf10BeGroups.pdf')
plt.show()
runs = 120
N_filtered = filtered_roi_cluster_data["counts"].sum()/runs

# Calculate the uncertainty as the square root of the sum of counts
uncertainty_filtered = np.sqrt(N_filtered)

# Print the counts and uncertainty
print(f"The sum of the filtered_roi_cluster_points for Be10 is: {N_filtered} ± {uncertainty_filtered}")
print(F"Confidence of the Silhouette Score of ROI: {SilhouetteScore_to_Confidence(filtered_silhouette_score)}")


for cluster_id in range(numberof_clusters):
    cluster_data = data[data['Cluster_KMeans'] == cluster_id]
    cluster_silhouette_score = cluster_data['silhouette_score'].mean()
    print(f"Group {cluster_id} --- Silhouette score {cluster_silhouette_score:.3f}")



silhouette_vals = silhouette_samples(X, data['Cluster_KMeans'])
y_lower, y_upper = 0, 0

plt.figure(figsize=(10, 6))
for i in range(numberof_clusters): 
    cluster_silhouette_vals = silhouette_vals[data['Cluster_KMeans'] == i]
    cluster_silhouette_vals.sort()
    y_upper += len(cluster_silhouette_vals)
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals, alpha=0.7, label=f"Cluster {i}")
    y_lower = y_upper

plt.axvline(x=np.mean(silhouette_vals), color="red", linestyle="--")
plt.xlabel("Silhouette Score")
plt.ylabel("Cluster Samples")
plt.title("Silhouette Plot for KMeans Clustering")
plt.legend()
plt.show()


num_runs = 100  # For each k value, we will run the clustering 'num_runs' times and average the inertia

# Range of clusters to test
k_range = range(1, 11)  # Test between 1 and 10 clusters
inertia_values = []
std_devs = []

# Run KMeans clustering for each k value
for k in k_range:
    inertia_list = []
    for _ in range(num_runs):
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=None)
        kmeans.fit(X)
        inertia_list.append(kmeans.inertia_)
    
    # Average inertia over multiple runs
    avg_inertia = np.mean(inertia_list)
    inertia_values.append(avg_inertia)
    
    # Calculate standard deviation (uncertainty) for each k
    std_dev = np.std(inertia_list)
    std_devs.append(std_dev)

# Plot the Elbow Curve with error bars
plt.figure(figsize=(8, 6))
plt.errorbar(k_range, inertia_values, yerr=std_devs, marker='o', linestyle='--', color='b', capsize=5)
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (Sum of Squared Distances)")
plt.xticks(k_range)
plt.grid(True)
plt.show()
