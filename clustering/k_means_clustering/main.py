import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

# Find optimal number of clusters using the Elbow method
wcss = []
for i in range(1, 11):
  k_means = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
  k_means.fit(X)
  wcss.append(k_means.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Apply K-Means using the optimal number of clusters
k_means = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_pred_clusters = k_means.fit_predict(X)
print(y_pred_clusters)

plt.scatter(X[y_pred_clusters==0, 0], X[y_pred_clusters==0, 1], s=100, c='Red', label='Careful')
plt.scatter(X[y_pred_clusters==1, 0], X[y_pred_clusters==1, 1], s=100, c='Blue', label='Standard')
plt.scatter(X[y_pred_clusters==2, 0], X[y_pred_clusters==2, 1], s=100, c='Green', label='Target')
plt.scatter(X[y_pred_clusters==3, 0], X[y_pred_clusters==3, 1], s=100, c='Cyan', label='Careless')
plt.scatter(X[y_pred_clusters==4, 0], X[y_pred_clusters==4, 1], s=100, c='Magenta', label='Sensible')
plt.scatter(k_means.cluster_centers_[:, 0], k_means.cluster_centers_[:, 1], s=300, c='Yellow', label='Centroids')
plt.title('Clusters of Clients')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()