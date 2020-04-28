import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)

y_pred = linear_regressor.predict(X_test)

plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, linear_regressor.predict(X_train), color='blue')
plt.title('Experience vs Salary (Training set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, linear_regressor.predict(X_test), color='blue')
plt.title('Experience vs Salary (Test set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()