import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
from sklearn.metrics import silhouette_samples, silhouette_score

# Define file path and subject
filepath = r'C:\Users\benja\Desktop\Speciale\Master-Thesis\Data\Første måling af Be10\2025_01_16_Benedikt\2025_01_16_Benedikt'
subject = "[CDAT0"

def list_files(directory):
    """List files in the directory, categorizing raw and other files."""
    raw_files = [file for file in os.listdir(directory) if file.endswith(".txt.mpa")]
    print("Raw files:", raw_files)
    return raw_files

def read_file(file_path, subject):
    """Read the file and extract header and data section index."""
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        if not lines:
            print("The file is empty")
            return None, None, None
        
        # Find header index
        header_index = next((idx for idx, line in enumerate(lines) if subject in line), None)
        if header_index is not None:
            print(f"Header index: {header_index}, Line: {lines[header_index]}")
        
        # Find data start index
        data_start = next((header_index + idx + 1 for idx, line in enumerate(lines[header_index:]) if line.startswith("[DATA]")), None)

        return lines, header_index, data_start
    except Exception as e:
        print(f"Error reading file: {e}")
        return None, None, None

def extract_data(lines, data_start):
    """Extract numerical data and return as a DataFrame."""
    if data_start is None:
        return pd.DataFrame()
    
    data = [line.strip().split() for line in lines[data_start:] if not any(c.isalpha() for c in line)]
    return pd.DataFrame(data, columns=["E_final", "dE", "counts"]).apply(pd.to_numeric)

def perform_clustering(data, n_clusters=4):
    """Perform KMeans clustering and calculate silhouette scores."""
    X = data[['E_final', 'dE']]
    kmeans = KMeans(n_clusters=n_clusters, random_state=30, init='k-means++')
    data['Cluster_KMeans'] = kmeans.fit_predict(X)

    # Silhouette scores
    data['silhouette_score'] = silhouette_samples(X, data['Cluster_KMeans'])

    return data, kmeans

def cluster_statistics(data, kmeans, cluster_id):
    """Compute statistics for a given cluster."""
    cluster_data = data[data['Cluster_KMeans'] == cluster_id]
    centroid = kmeans.cluster_centers_[cluster_id]
    
    cluster_size = len(cluster_data)
    mean_silhouette = cluster_data['silhouette_score'].mean()
    
    # Compute distances
    distances = np.linalg.norm(cluster_data[['E_final', 'dE']].values - centroid, axis=1)
    mean_distance = distances.mean()
    variance_distance = distances.var()
    
    print(f"Cluster {cluster_id} - Size: {cluster_size}, Avg Silhouette: {mean_silhouette:.4f}")
    print(f"Mean Distance to Centroid: {mean_distance:.4f}, Variance: {variance_distance:.4f}")
    
    return cluster_data, centroid

def filter_cluster(cluster_data, centroid, radius=35):
    """Filter cluster points within a defined radius from the centroid."""
    distances = np.linalg.norm(cluster_data[['E_final', 'dE']].values - centroid, axis=1)
    filtered_data = cluster_data[distances <= radius]
    return filtered_data

def plot_clusters(data, kmeans, filtered_data=None, radius=35):
    """Plot clusters with convex hulls and highlight filtered points."""
    fig, ax = plt.subplots(figsize=(8, 6))
    palette = sns.color_palette("viridis", n_colors=kmeans.n_clusters)
    
    for cluster_id in data['Cluster_KMeans'].unique():
        cluster_points = data[data['Cluster_KMeans'] == cluster_id]
        ax.scatter(cluster_points['E_final'], cluster_points['dE'], color=palette[cluster_id], s=2, label=f'Cluster {cluster_id}')

        if len(cluster_points) >= 3:
            hull = ConvexHull(cluster_points[['E_final', 'dE']].values)
            hull_points = cluster_points.iloc[hull.vertices]
            ax.fill(hull_points['E_final'], hull_points['dE'], color=palette[cluster_id], alpha=0.3)

    if filtered_data is not None:
        ax.scatter(filtered_data['E_final'], filtered_data['dE'], color='red', s=10, label='Filtered Points')

        circle = plt.Circle((filtered_data['E_final'].mean(), filtered_data['dE'].mean()), radius, color='blue', fill=False, linestyle='--', linewidth=2)
        ax.add_patch(circle)
    
    ax.set_xlabel('E_final [keV]')
    ax.set_ylabel('dE [keV]')
    ax.set_title('Clustered Data with Outlines')
    ax.legend()
    plt.tight_layout()
    plt.show()

# Main script
list_of_raw_files = list_files(filepath)
if list_of_raw_files:
    file_path = os.path.join(filepath, list_of_raw_files[0])
    lines, header_index, data_start = read_file(file_path, subject)

    if lines:
        data = extract_data(lines, data_start)
        print(data.head())

        data, kmeans = perform_clustering(data)
        cluster_data, cluster_centroid = cluster_statistics(data, kmeans, cluster_id=2)

        filtered_cluster = filter_cluster(cluster_data, cluster_centroid)
        plot_clusters(data, kmeans, filtered_cluster)
