import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


np.random.seed(42)
X = np.random.rand(100, 1) * 10 
y = 2.5 * X + np.random.randn(100, 1) * 2  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


r2 = r2_score(y_test, y_pred)


n = X_test.shape[0]  
p = X_test.shape[1]  
adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)


mae = mean_absolute_error(y_test, y_pred)


mse = mean_squared_error(y_test, y_pred)


print(f"R-Squared: {r2}")
print(f"Adjusted R-Squared: {adjusted_r2}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
