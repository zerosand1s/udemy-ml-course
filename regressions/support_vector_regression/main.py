import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1, 1))

regressor = SVR(kernel='rbf')
regressor.fit(X, y)

salary_using_svr = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))
print(salary_using_svr)

# plt.scatter(X, y, color='red')
# plt.plot(X, regressor.predict(X), color='blue')
# plt.title('Experience vs Salary (SVR)')
# plt.xlabel('Years of experience')
# plt.ylabel('Salary')
# plt.show()