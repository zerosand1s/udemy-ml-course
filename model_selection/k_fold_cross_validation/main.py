import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

kernel_svm_classifier = SVC(kernel='rbf', random_state=0)
kernel_svm_classifier.fit(X_train, y_train)

y_pred = kernel_svm_classifier.predict(X_test)
print(y_pred)

cm = confusion_matrix(y_test, y_pred)
print(cm)

# Applying K Fold Cross validation
accuracies = cross_val_score(estimator=kernel_svm_classifier, X=X_train, y=y_train, cv=10)
print(accuracies)
print(accuracies.mean())
print(accuracies.std())