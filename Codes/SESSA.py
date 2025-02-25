import pandas as pd
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import gaussian_kde

# Load data once at the start
tidy = pd.read_csv("Data/med_events.csv")
tidy.columns = ["pnr", "eksd", "perday", "ATC", "dur_original"]
tidy['eksd'] = pd.to_datetime(tidy['eksd'], format='%m/%d/%Y')

def plot_clustering_analysis(data_scaled, method, arg1, dfper, labels, x_vals, y_vals):
    """
    Common plotting function for both methods: ECDF, Density, and Silhouette
    """
    plt.figure(figsize=(15, 5))
    
    # Plot ECDF
    plt.subplot(1, 3, 1)
    plt.plot(x_vals, y_vals)
    plt.title(f"ECDF - {arg1} ({method})")
    plt.xlabel("Interval (days)")
    plt.ylabel("Cumulative Probability")
    plt.grid(True, alpha=0.3)
    
    # Plot Density
    plt.subplot(1, 3, 2)
    log_intervals = np.log(dfper['x'])
    kde = gaussian_kde(log_intervals)
    x_range = np.linspace(log_intervals.min(), log_intervals.max(), 100)
    plt.plot(x_range, kde(x_range))
    plt.title(f'Density of log(event_interval) ({arg1}) - {method}')
    plt.xlabel('log(event_interval)')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)
    
    # Plot Silhouette
    plt.subplot(1, 3, 3)
    if method == "K-means":
        range_n_clusters = range(2, 11)
        silhouette_scores = []
        for n_clusters in range_n_clusters:
            kmeans = KMeans(n_clusters=n_clusters, random_state=1234)
            cluster_labels = kmeans.fit_predict(data_scaled)
            silhouette_avg = silhouette_score(data_scaled, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        plt.plot(range_n_clusters, silhouette_scores, marker='o')
    else:  # DBSCAN
        # Calculate silhouette score for DBSCAN result
        if len(set(labels)) > 1 and -1 not in labels:  # Only if we have valid clusters
            silhouette_avg = silhouette_score(data_scaled, labels)
            plt.axhline(y=silhouette_avg, color='r', linestyle='--',
                       label=f'Silhouette Score: {silhouette_avg:.3f}')
            plt.legend()
    
    plt.title(f"Silhouette Analysis - {arg1} ({method})")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return silhouette_scores if method == "K-means" else None

def See(arg1="medA"):
    # Filter data for specific medication
    drug_see = tidy[tidy['ATC'] == arg1].copy()
    
    # Calculate intervals between prescriptions for each patient
    drug_see['prev_eksd'] = drug_see.groupby('pnr')['eksd'].shift(1)
    drug_see = drug_see.dropna(subset=['prev_eksd'])
    
    # Sample one interval per patient (simpler version)
    drug_see_sample = pd.DataFrame()
    for _, group in drug_see.groupby('pnr'):
        drug_see_sample = pd.concat([drug_see_sample, group.sample(n=1)])
    
    # Calculate interval in days
    drug_see_sample['event_interval'] = (
        drug_see_sample['eksd'] - drug_see_sample['prev_eksd']
    ).dt.days
    
    # Generate ECDF
    intervals = np.sort(drug_see_sample['event_interval'])
    n = len(intervals)
    y_vals = np.arange(1, n + 1) / n
    x_vals = intervals
    
    # Prepare data for clustering
    dfper = pd.DataFrame({'x': x_vals, 'y': y_vals})
    dfper = dfper[dfper['y'] <= 0.8]  # Keep only 80% of data
    
    # Scale the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(dfper)
    
    # Calculate silhouette scores first
    range_n_clusters = range(2, 11)
    silhouette_scores = []
    labels = None
    
    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=1234)
        cluster_labels = kmeans.fit_predict(data_scaled)
        silhouette_avg = silhouette_score(data_scaled, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        
        # Save the labels for the optimal number of clusters
        if n_clusters == range_n_clusters[np.argmax(silhouette_scores)]:
            labels = cluster_labels
    
    dfper['cluster'] = labels
    
    # Plot results
    plot_clustering_analysis(
        data_scaled, "K-means", arg1, dfper, labels, x_vals, y_vals
    )
    
    return max(x_vals[y_vals <= 0.8]), dfper

def see_assumption(data, title=""):
    # Calculate prescription order and intervals
    data = data.copy()
    data['prev_eksd'] = data.groupby('pnr')['eksd'].shift(1)
    data['Duration'] = (data['eksd'] - data['prev_eksd']).dt.days
    
    # Create prescription number (p_number) for each patient
    data['p_number'] = data.groupby('pnr').cumcount() + 1
    
    # Filter to prescriptions >= 2 (since first prescription has no interval)
    data = data[data['p_number'] >= 2]
    
    # Calculate median duration for each patient
    medians = data.groupby('pnr')['Duration'].median()
    overall_median = medians.median()
    
    # Create boxplot
    plt.figure(figsize=(10, 6))
    bp = plt.boxplot([group['Duration'] for name, group in data.groupby('p_number')])
    
    # Add horizontal line for median of medians
    plt.axhline(y=overall_median, color='r', linestyle='--', label='Median of patient medians')
    
    plt.title(f'Duration by Prescription Number - {title}')
    plt.xlabel('Prescription Number')
    plt.ylabel('Duration (days)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return plt

def See_dbscan(arg1="medA"):
    # Filter data for specific medication
    drug_see = tidy[tidy['ATC'] == arg1].copy()
    
    # Calculate intervals between prescriptions for each patient
    drug_see['prev_eksd'] = drug_see.groupby('pnr')['eksd'].shift(1)
    drug_see = drug_see.dropna(subset=['prev_eksd'])
    
    # Sample one interval per patient (simpler version)
    drug_see_sample = pd.DataFrame()
    for _, group in drug_see.groupby('pnr'):
        drug_see_sample = pd.concat([drug_see_sample, group.sample(n=1)])
    
    # Calculate interval in days
    drug_see_sample['event_interval'] = (
        drug_see_sample['eksd'] - drug_see_sample['prev_eksd']
    ).dt.days
    
    # Generate ECDF
    intervals = np.sort(drug_see_sample['event_interval'])
    n = len(intervals)
    y_vals = np.arange(1, n + 1) / n
    x_vals = intervals
    
    # Retain only 80% of the ECDF
    mask_80 = y_vals <= 0.8
    x_80 = x_vals[mask_80]
    y_80 = y_vals[mask_80]
    
    # Plot ECDFs
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(x_80, y_80)
    plt.title(f"80% ECDF - {arg1}")
    plt.xlabel("Interval (days)")
    plt.ylabel("Cumulative Probability")
    
    plt.subplot(1, 2, 2)
    plt.plot(x_vals, y_vals)
    plt.title(f"100% ECDF - {arg1}")
    plt.xlabel("Interval (days)")
    plt.ylabel("Cumulative Probability")
    
    plt.tight_layout()
    plt.show()
    
    # Prepare data for clustering
    dfper = pd.DataFrame({'x': x_vals, 'y': y_vals})
    dfper = dfper[dfper['y'] <= 0.8]  # Keep only 80% of data
    
    # Scale the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(dfper)
    
    # Find optimal epsilon using nearest neighbors
    neighbors = NearestNeighbors(n_neighbors=2)
    neighbors_fit = neighbors.fit(data_scaled)
    distances, indices = neighbors_fit.kneighbors(data_scaled)
    distances = np.sort(distances[:, 1])
    
    # Plot k-distance graph
    plt.figure()
    plt.plot(range(len(distances)), distances)
    plt.title(f'K-distance Graph - {arg1}')
    plt.xlabel('Points sorted by distance')
    plt.ylabel('Distance to nearest neighbor')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Use elbow method to find epsilon
    knee = np.diff(distances, 2)
    epsilon = distances[np.argmax(knee)]
    
    # Apply DBSCAN
    dbscan = DBSCAN(eps=epsilon, min_samples=3)
    dfper['cluster'] = dbscan.fit_predict(data_scaled)
    
    # Plot clustered data
    plt.figure()
    # Plot noise points (-1) in black
    noise = dfper[dfper['cluster'] == -1]
    plt.scatter(noise['x'], noise['y'], c='black', label='Noise', alpha=0.5)
    
    # Plot clusters
    for cluster in sorted(set(dfper['cluster'][dfper['cluster'] != -1])):
        cluster_data = dfper[dfper['cluster'] == cluster]
        plt.scatter(cluster_data['x'], cluster_data['y'], 
                   label=f'Cluster {cluster}', alpha=0.7)
    
    plt.title(f'DBSCAN Clustering Results - {arg1}')
    plt.xlabel('Interval (days)')
    plt.ylabel('Cumulative Probability')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Use same common plotting function
    plot_clustering_analysis(
        data_scaled, "DBSCAN", arg1, dfper, dfper['cluster'], x_vals, y_vals
    )
    
    # Print clustering statistics
    n_clusters = len(set(dfper['cluster'])) - (1 if -1 in dfper['cluster'] else 0)
    n_noise = list(dfper['cluster']).count(-1)
    print(f"\nDBSCAN Results for {arg1}:")
    print(f"Estimated number of clusters: {n_clusters}")
    print(f"Estimated number of noise points: {n_noise}")
    print(f"Epsilon used: {epsilon:.3f}")
    
    return max(x_80), dfper

def compare_clustering_methods(medA_kmeans, medA_dbscan, medB_kmeans, medB_dbscan):
    """
    Compare and print insights between K-means and DBSCAN results
    """
    print("\nComparison of Clustering Methods for Sessa Empirical Estimator:")
    print("-" * 60)
    
    # Compare number of clusters
    print("\nNumber of Clusters:")
    print(f"MedA - K-means: {len(set(medA_kmeans['cluster']))}")
    print(f"MedA - DBSCAN: {len(set(medA_dbscan['cluster'])) - (1 if -1 in medA_dbscan['cluster'] else 0)}")
    print(f"MedB - K-means: {len(set(medB_kmeans['cluster']))}")
    print(f"MedB - DBSCAN: {len(set(medB_dbscan['cluster'])) - (1 if -1 in medB_dbscan['cluster'] else 0)}")
    
    # Compare noise points (DBSCAN only)
    print("\nNoise Points (DBSCAN only):")
    print(f"MedA: {sum(medA_dbscan['cluster'] == -1)} points ({(sum(medA_dbscan['cluster'] == -1)/len(medA_dbscan))*100:.1f}%)")
    print(f"MedB: {sum(medB_dbscan['cluster'] == -1)} points ({(sum(medB_dbscan['cluster'] == -1)/len(medB_dbscan))*100:.1f}%)")
    
    # Compare cluster sizes
    print("\nCluster Size Distribution:")
    print("MedA - K-means:", dict(zip(*np.unique(medA_kmeans['cluster'], return_counts=True))))
    print("MedA - DBSCAN:", dict(zip(*np.unique(medA_dbscan['cluster'], return_counts=True))))
    print("MedB - K-means:", dict(zip(*np.unique(medB_kmeans['cluster'], return_counts=True))))
    print("MedB - DBSCAN:", dict(zip(*np.unique(medB_dbscan['cluster'], return_counts=True))))

# Create boxplots for individual medications and all medications combined
print("\nBoxplots for medA:")
medA_data = tidy[tidy['ATC'] == "medA"].copy()
see_assumption(medA_data, "MedA")

print("\nBoxplots for medB:")
medB_data = tidy[tidy['ATC'] == "medB"].copy()
see_assumption(medB_data, "MedB")

print("\nBoxplots for All Medications Combined:")
see_assumption(tidy, "All Medications")

# Run the clustering comparisons
print("\nRunning clustering analysis...")
max_interval_medA_kmeans, clusters_medA_kmeans = See("medA")
max_interval_medB_kmeans, clusters_medB_kmeans = See("medB")

print("\nRunning DBSCAN analysis...")
max_interval_medA_dbscan, clusters_medA_dbscan = See_dbscan("medA")
max_interval_medB_dbscan, clusters_medB_dbscan = See_dbscan("medB")

# Compare results
compare_clustering_methods(
    clusters_medA_kmeans,
    clusters_medA_dbscan,
    clusters_medB_kmeans,
    clusters_medB_dbscan
)

# Print key insights
print("\nKey Insights:")
print("-" * 60)
print("""
1. Algorithm Characteristics:
   - K-means forces all points into clusters, potentially over-fitting noise
   - DBSCAN identifies noise points, providing more realistic clustering in sparse areas

2. Maximum Intervals:
   MedA: K-means: {:.1f} days, DBSCAN: {:.1f} days
   MedB: K-means: {:.1f} days, DBSCAN: {:.1f} days

3. Clustering Behavior:
   - K-means creates more evenly-sized clusters
   - DBSCAN finds naturally occurring dense regions

4. Practical Implications:
   - K-means might be better for regular prescription patterns
   - DBSCAN might be better for identifying irregular prescriptions
""".format(
    max_interval_medA_kmeans, max_interval_medA_dbscan,
    max_interval_medB_kmeans, max_interval_medB_dbscan
))