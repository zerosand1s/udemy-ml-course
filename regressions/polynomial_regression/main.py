import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

linear_regressor_one = LinearRegression()
linear_regressor_one.fit(X, y)

polynomial_regressor = PolynomialFeatures(degree=4)
X_poly = polynomial_regressor.fit_transform(X)

linear_regressor_two = LinearRegression()
linear_regressor_two.fit(X_poly, y)

salary_using_linear_regression = linear_regressor_one.predict([[6.5]])
print(salary_using_linear_regression)

salary_using_polynomial_regression = linear_regressor_two.predict(polynomial_regressor.fit_transform([[6.5]]))
print(salary_using_polynomial_regression)

# plt.scatter(X, y, color='red')
# plt.plot(X, linear_regressor_one.predict(X), color='blue')
# plt.title('Experience vs Salary (Linear Regression)')
# plt.xlabel('Years of experience')
# plt.ylabel('Salary')
# plt.show()

# X_grid = np.arange(min(X), max(X), 0.1)
# X_grid = X_grid.reshape((len(X_grid), 1))
# plt.scatter(X, y, color='red')
# plt.plot(X_grid, linear_regressor_two.predict(polynomial_regressor.fit_transform(X_grid)), color='blue')
# plt.title('Experience vs Salary (Polynomial Regression)')
# plt.xlabel('Years of experience')
# plt.ylabel('Salary')
# plt.show()