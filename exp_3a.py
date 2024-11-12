import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, LeaveOneOut, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt



df = pd.read_excel('/content/Real_estate_valuation_data_set.xlsx')
print(df)

df.shape

df.describe()

df.info()

df.isnull().sum()

#define features and target
x = df.iloc[:,:-1]
y = df.iloc[:, -1]

#splitting into ttraining and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)



lr = LinearRegression()
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Linear Regression Performance:")
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R-squared: {r2}")

loo = LeaveOneOut()
kf = KFold(n_splits=5, shuffle = True, random_state = 42)

loo_scores = cross_val_score(lr,x,y,cv = loo, scoring = 'neg_mean_squared_error')
loo_mse = -loo_scores.mean()

kf_scores = cross_val_score(lr,x,y,cv = kf, scoring = 'neg_mean_squared_error')
kf_mse = -kf_scores.mean()

print()

print(f"Leave-One-Out Cross-Validation MSE: {loo_mse}")
print(f"Five-Fold Cross-Validation MSE: {kf_mse}")

print(f"Comparision of Evaluation Methods:")
print(f"Train-Test Split MSE: {mse}")
print(f"Leave-One-Out Cross-Validation MSE: {loo_mse}")
print(f"Five-Fold Cross-Validation MSE: {kf_mse}")

"""L2 regurallization[link text](https://)(ridge reguralization)"""

ridge = Ridge(alpha = 1.0)
ridge.fit(x_train, y_train)
ridge_pred = ridge.predict(x_test)

ridge_mse = mean_squared_error(y_test, ridge_pred)
ridge_mae = mean_absolute_error(y_test, ridge_pred)
ridge_r2 = r2_score(y_test, ridge_pred)

print(f"Ridge Regression Performance:")
print(f"Mean Squared Error: {ridge_mse}")
print(f"Mean Absolute Error: {ridge_mae}")
print(f"R2 Score: {ridge_r2}")

"""l2 reguralization (lassooo)"""

#model
lasso = Lasso(alpha = 1.0)
lasso.fit(x_train, y_train)
lasso_pred = lasso.predict(x_test)

#comparing with the data sets
lasso_mse = mean_squared_error(y_test, lasso_pred)
lasso_mae = mean_absolute_error(y_test, lasso_pred)
lasso_r2 = r2_score(y_test, lasso_pred)

print(f"Lasso Regression Performance:")
print(f"Mean Squared Error: {lasso_mse}")
print(f"Mean Absolute Error: {lasso_mae}")
print(f"R2 Score: {lasso_r2}")

print("Key Findings:")
print(f"Linear Regression MSE: {mse}")
print(f"Ridge Regression MSE: {ridge_mse}")
print(f"Lasso Regression MSE: {lasso_mse}")