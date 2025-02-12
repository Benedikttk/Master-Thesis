import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
from sklearn.metrics import silhouette_samples
from sklearn.metrics import silhouette_score

# Define file path and subject
filepath = r'C:\Users\benja\Desktop\Speciale\Master-Thesis\Data\Første måling af Be10\2025_01_16_Benedikt\2025_01_16_Benedikt'
subject = "[CDAT0"

# List files in the directory
for file in os.listdir(filepath):
    if file.startswith("Be"):
        print(file)
    else:
        print(f"The other files: {file}")
print("\n")

# Get list of raw files
list_of_raw_files = [file for file in os.listdir(filepath) if file.endswith(".txt.mpa")]
print("These are the raw files", list_of_raw_files)

# Read the file
file_path = os.path.join(filepath, list_of_raw_files[1])
with open(file_path, 'r') as file:
    lines = file.readlines()

if not lines:
    print("The file is empty")
else:
    print("The file is not empty")

# Find header index
header_index = next((idx for idx, line in enumerate(lines) if subject in line), None)
if header_index is not None:
    print(f"Printing the header index of {subject}:", header_index)
    print("The line corresponding to header_index is:", lines[header_index])
else:
    print("No matching header found.")

# Find data index
data_index = next((idx for idx, line in enumerate(lines[header_index:]) if line.startswith("[DATA]")), None)
if data_index is not None:
    data_start = header_index + data_index + 1
else:
    print("No data section found.")
    data_start = None

# Extract data
data = []
if data_start is not None:
    for line in lines[data_start:]:
        if any(char.isalpha() for char in line):
            break
        data.append(line.strip().split())

# Create DataFrame
data = pd.DataFrame(data, columns=["E_final", "dE", "counts"]).apply(pd.to_numeric)
print(data.head())

numberof_clusters = 4  # optimal 
# KMeans clustering
X = data[['E_final', 'dE']]
kmeans = KMeans(n_clusters=numberof_clusters, random_state=30, init='k-means++')
data['Cluster_KMeans'] = kmeans.fit_predict(X)
print(data[['E_final', 'dE', 'Cluster_KMeans']].head())

# Silhouette scores
data['silhouette_score'] = silhouette_samples(X, data['Cluster_KMeans'])
cluster_2_data = data[data['Cluster_KMeans'] == 2]
cluster_2_silhouette_score = cluster_2_data['silhouette_score'].mean()
print(f"Average Silhouette Score for Cluster 2: {cluster_2_silhouette_score}")

# Distance to centroid
cluster_2_centroid = kmeans.cluster_centers_[2]
cluster_2_data['distance_to_centroid'] = np.linalg.norm(cluster_2_data[['E_final', 'dE']].values - cluster_2_centroid, axis=1)
mean_distance_to_centroid = cluster_2_data['distance_to_centroid'].mean()
print(f"Mean Distance to Centroid for Cluster 2: {mean_distance_to_centroid}")

# Distance to other centroids
distances_to_other_centroids = np.linalg.norm(kmeans.cluster_centers_ - cluster_2_centroid, axis=1)
print(f"Distance of Cluster 2's centroid to other centroids: {distances_to_other_centroids}")

# Number of points in Cluster 2
cluster_2_size = cluster_2_data.shape[0]
print(f"Number of points in Cluster 2: {cluster_2_size}")

# Plot clusters with Convex Hull
fig, ax = plt.subplots(figsize=(8, 6))
for cluster_id in data['Cluster_KMeans'].unique():
    cluster_points = data[data['Cluster_KMeans'] == cluster_id]
    points = cluster_points[['E_final', 'dE']].values
    if len(points) >= 3:
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        ax.fill(hull_points[:, 0], hull_points[:, 1], color=sns.color_palette("viridis", n_colors=numberof_clusters)[cluster_id], alpha=0.3)
        ax.plot(hull_points[:, 0], hull_points[:, 1], color=sns.color_palette("viridis", n_colors=numberof_clusters)[cluster_id], lw=2)
    ax.scatter(cluster_points['E_final'], cluster_points['dE'], c=[sns.color_palette("viridis", n_colors=numberof_clusters)[cluster_id]] * len(cluster_points), label=f'Cluster {cluster_id}', s=2)
ax.set_xlabel('E_final [keV]')
ax.set_ylabel('dE [keV]')
ax.set_title('Clustered Data with Outlines')
ax.legend()
plt.tight_layout()
plt.show()

# Generate lists of points for cluster 2
cluster_2_points = data[data['Cluster_KMeans'] == 2]
x_values = cluster_2_points['E_final'].tolist()
y_values = cluster_2_points['dE'].tolist()

#print(f"Cluster 2 - X values: {x_values}")
#print(f"Cluster 2 - Y values: {y_values}")

# Cleaning the cluster now
cluster_2_radius = 35  # radius for filtering
circle = plt.Circle((cluster_2_centroid[0], cluster_2_centroid[1]), cluster_2_radius, color='blue', fill=False, linestyle='--', linewidth=2)

# Calculate the Euclidean distance from each point to the centroid of Cluster 2
distances = np.linalg.norm(cluster_2_points[['E_final', 'dE']].values - cluster_2_centroid, axis=1)

# Filter the points that are within the radius
filtered_cluster_2_points = cluster_2_points[distances <= cluster_2_radius]

# Now filtered_cluster_2_points contains the points within the circle
filtered_x_values = filtered_cluster_2_points['E_final'].tolist()
filtered_y_values = filtered_cluster_2_points['dE'].tolist()

#print(f"Filtered Cluster 2 - X values within radius: {filtered_x_values}")
#print(f"Filtered Cluster 2 - Y values within radius: {filtered_y_values}")

# Plot the filtered points
fig, ax = plt.subplots(figsize=(8, 6))
for cluster_id in data['Cluster_KMeans'].unique():
    cluster_points = data[data['Cluster_KMeans'] == cluster_id]
    points = cluster_points[['E_final', 'dE']].values
    if len(points) >= 3:
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        ax.fill(hull_points[:, 0], hull_points[:, 1], color=sns.color_palette("viridis", n_colors=numberof_clusters)[cluster_id], alpha=0.3)
        ax.plot(hull_points[:, 0], hull_points[:, 1], color=sns.color_palette("viridis", n_colors=numberof_clusters)[cluster_id], lw=2)

    ax.scatter(cluster_points['E_final'], cluster_points['dE'], c=[sns.color_palette("viridis", n_colors=numberof_clusters)[cluster_id]] * len(cluster_points), label=f'Cluster {cluster_id}', s=2)

# Add the circle
ax.add_patch(circle)

# Plot the filtered points within the circle in red
ax.scatter(filtered_cluster_2_points['E_final'], filtered_cluster_2_points['dE'], color='red', label='Filtered Cluster 2 Points', s=20)

# Set labels and title
ax.set_xlabel('E_final [keV]')
ax.set_ylabel('dE [keV]')
ax.set_title('Filtered Cluster 2 Points within Circle')
ax.legend()
plt.tight_layout()
plt.show()

# Re-cluster the filtered points (Cluster 2 points within the radius)
filtered_X = filtered_cluster_2_points[['E_final', 'dE']]

# Apply KMeans to the filtered data to create at least two clusters
kmeans_filtered = KMeans(n_clusters=2, random_state=30, init='k-means++')
filtered_cluster_2_points['Filtered_Cluster'] = kmeans_filtered.fit_predict(filtered_X)

# Now, calculate the silhouette score for the re-clustered points
filtered_silhouette_score = silhouette_score(filtered_X, filtered_cluster_2_points['Filtered_Cluster'])

print(f"Silhouette Score for the Filtered Cluster 2 (after re-clustering): {filtered_silhouette_score}")

# Original Centroid (before filtering)
original_centroid = kmeans.cluster_centers_[2]  # Assuming Cluster 2 is the target

# Calculate distances of all points in the original cluster to the centroid
distances_original = np.linalg.norm(cluster_2_data[['E_final', 'dE']].values - original_centroid, axis=1)

# Mean Distance to Centroid (Original)
mean_distance_original = np.mean(distances_original)

# Variance (Spread) of Points around the Centroid (Original)
variance_original = np.var(distances_original)

# Calculate standard error for mean distance and variance for original cluster
se_mean_distance_original = np.std(distances_original) / np.sqrt(len(distances_original))
se_variance_original = np.sqrt((2 * variance_original ** 2) / (len(distances_original) - 1))

# Output the values
print(f"Mean Distance to Centroid (Original): {mean_distance_original:.4f} ± {se_mean_distance_original:.4f}")
print(f"Variance of Distances (Original): {variance_original:.4f} ± {se_variance_original:.4f}")

# Filtered Centroid (after filtering)
filtered_centroid = filtered_cluster_2_points[['E_final', 'dE']].mean().values  # Mean of the filtered points

# Calculate distances of all filtered points to the new centroid
distances_filtered = np.linalg.norm(filtered_cluster_2_points[['E_final', 'dE']].values - filtered_centroid, axis=1)

# Mean Distance to Centroid (Filtered)
mean_distance_filtered = np.mean(distances_filtered)

# Variance (Spread) of Points around the Centroid (Filtered)
variance_filtered = np.var(distances_filtered)

# Calculate standard error for filtered cluster distances
se_mean_distance_filtered = np.std(distances_filtered) / np.sqrt(len(distances_filtered))
se_variance_filtered = np.sqrt((2 * variance_filtered ** 2) / (len(distances_filtered) - 1))

# Output the values for filtered cluster
print(f"Mean Distance to Centroid (Filtered): {mean_distance_filtered:.4f} ± {se_mean_distance_filtered:.4f}")
print(f"Variance of Distances (Filtered): {variance_filtered:.4f} ± {se_variance_filtered:.4f}")
