import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Apply K-PCA
k_pca = KernelPCA(n_components=2, kernel='rbf')
X_train = k_pca.fit_transform(X_train)
X_test = k_pca.transform(X_test)

logistic_classifier = LogisticRegression(random_state=0)
logistic_classifier.fit(X_train, y_train)

y_pred = logistic_classifier.predict(X_test)
print(y_pred)

cm = confusion_matrix(y_test, y_pred)
print(cm)