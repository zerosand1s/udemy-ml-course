import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

dataset = pd.read_csv('Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

ct = ColumnTransformer(transformers=[('encode', OneHotEncoder(), [3])], remainder='passthrough')
X = ct.fit_transform(X)

# Removing one dummy variable (Most libraries will take of this)
X = X[:, 1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)

y_pred = linear_regressor.predict(X_test)

# Backword elimination
X = np.append(arr=np.ones((X.shape[0], 1)).astype(int), values=X, axis=1)
X_optimal = np.array(X[:, [0, 1, 2, 3, 4, 5]], dtype=float)
ols_regressor = sm.OLS(endog=y, exog=X_optimal).fit()
print(ols_regressor.summary())

X_optimal = np.array(X[:, [0, 1, 3, 4, 5]], dtype=float)
ols_regressor = sm.OLS(endog=y, exog=X_optimal).fit()
print(ols_regressor.summary())

X_optimal = np.array(X[:, [0, 3, 4, 5]], dtype=float)
ols_regressor = sm.OLS(endog=y, exog=X_optimal).fit()
print(ols_regressor.summary())

X_optimal = np.array(X[:, [0, 3, 5]], dtype=float)
ols_regressor = sm.OLS(endog=y, exog=X_optimal).fit()
print(ols_regressor.summary())

X_optimal = np.array(X[:, [0, 3]], dtype=float)
ols_regressor = sm.OLS(endog=y, exog=X_optimal).fit()
print(ols_regressor.summary())