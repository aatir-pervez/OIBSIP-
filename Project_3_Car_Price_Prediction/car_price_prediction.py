# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv("car_data.csv")

# Display first 5 rows
print("First 5 rows:\n")
print(data.head())

# Check dataset info
print("\nDataset Info:\n")
print(data.info())

# Check missing values
print("\nMissing Values:\n")
print(data.isnull().sum())

# Drop unnecessary column
data = data.drop(['Car_Name'], axis=1)

# Convert categorical columns into numerical using one-hot encoding
data = pd.get_dummies(data, drop_first=True)

print("\nAfter Encoding:\n")
print(data.head())

# Define features and target
X = data.drop('Selling_Price', axis=1)
y = data['Selling_Price']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

from sklearn.metrics import r2_score, mean_absolute_error

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\nModel Evaluation:")
print("R2 Score:", r2)
print("Mean Absolute Error:", mae)

# Plot Actual vs Predicted
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")
plt.show()