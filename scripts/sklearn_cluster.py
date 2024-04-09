#!/usr/local/bin/python

# Import packages
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Define the number of homes
n_samples = 500

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data for each feature
residential_sales_price = np.random.randint(50000, 500000, size=n_samples)
residential_sales_variance = np.random.uniform(0, 0.2, size=n_samples)
mortgage_foreclosure_filings = np.random.randint(0, 50, size=n_samples)
parcel_year_built = np.random.randint(1900, 2020, size=n_samples)
parcel_condition = np.random.randint(1, 5, size=n_samples)  # Assuming condition rating from 1 to 5
vacant_lot_area = np.random.uniform(1000, 5000, size=n_samples)
building_violations = np.random.randint(0, 10, size=n_samples)
owner_occupancy = np.random.uniform(0, 100, size=n_samples)  # Percentage of owner occupancy
subsidized_housing_units = np.random.randint(0, 50, size=n_samples)

# Create a DataFrame to store the synthetic data
data = pd.DataFrame({
    'Residential Sales Price': residential_sales_price,
    'Residential Sales Variance': residential_sales_variance,
    'Mortgage Foreclosure Filings': mortgage_foreclosure_filings,
    'Parcel Year Built': parcel_year_built,
    'Parcel Condition': parcel_condition,
    'Vacant Lot Area': vacant_lot_area,
    'Building Violations': building_violations,
    'Owner Occupancy': owner_occupancy,
    'Subsidized Housing Units': subsidized_housing_units
})


# Standardize the features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Choose the number of clusters
n_clusters = 5

# Initialize KMeans
kmeans = KMeans(n_clusters=n_clusters, random_state=42)

# Fit the KMeans model to the scaled data
kmeans.fit(scaled_data)

# Get cluster labels
cluster_labels = kmeans.labels_

# Add cluster labels to the original dataset
data['Cluster'] = cluster_labels

# Analyze characteristics of each cluster
cluster_summary = data.groupby('Cluster').mean()
print(cluster_summary)

import matplotlib.pyplot as plt

# Visualize clusters (example for 2D data)
plt.figure(figsize=(10, 6))

# Plot the data points with cluster labels
for cluster in range(n_clusters):
    cluster_data = scaled_data[cluster_labels == cluster]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {cluster+1}')

# Plot the centroids
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, color='black', label='Centroids')

plt.title('KMeans Clustering')
plt.xlabel('Feature 1 (Standardized)')
plt.ylabel('Feature 2 (Standardized)')
plt.legend()
plt.grid(True)
plt.show()