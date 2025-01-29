import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Sample purchase history data
data = {
    'CustomerID': [1, 2, 3, 4, 5],
    'TotalSpend': [200, 400, 300, 500, 250],
    'Frequency': [10, 15, 13, 20, 12]
}

# Creating DataFrame
df = pd.DataFrame(data)

# Standardizing the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[['TotalSpend', 'Frequency']])

# K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(scaled_features)
df['Cluster'] = kmeans.labels_

# Plotting the clusters
plt.scatter(df['TotalSpend'], df['Frequency'], c=df['Cluster'])
plt.xlabel('Total Spend')
plt.ylabel('Frequency')
plt.title('Customer Clusters based on Purchase History')
plt.show()
