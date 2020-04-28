import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

# Using Dendrogram to find number of optimal clusters
# dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
# plt.title('Dendrogram')
# plt.xlabel('Customers')
# plt.ylabel('Euclidean Distances')
# plt.show()

# Apply HC using the optimal number of clusters
hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(X)
print(y_hc)

plt.scatter(X[y_hc==0, 0], X[y_hc==0, 1], s=100, c='Red', label='Careful')
plt.scatter(X[y_hc==1, 0], X[y_hc==1, 1], s=100, c='Blue', label='Standard')
plt.scatter(X[y_hc==2, 0], X[y_hc==2, 1], s=100, c='Green', label='Target')
plt.scatter(X[y_hc==3, 0], X[y_hc==3, 1], s=100, c='Cyan', label='Careless')
plt.scatter(X[y_hc==4, 0], X[y_hc==4, 1], s=100, c='Magenta', label='Sensible')
plt.title('Clusters of Clients')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()