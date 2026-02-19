# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("Unemployment_Project/Unemployment in India.csv")

# Display first 5 rows
print("First 5 rows of dataset:\n")
print(data.head())

# Display basic information
print("\nDataset Information:\n")
print(data.info())

# Check for missing values
print("\nMissing Values:\n")
print(data.isnull().sum())

# Remove extra spaces from column names
data.columns = data.columns.str.strip()

# Drop missing values
data = data.dropna()

# Display column names
print("\nColumn Names:\n")
print(data.columns)

# Convert Date column to datetime format
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)

# Sort data by Date
data = data.sort_values(by='Date')

print("\nSummary Statistics:\n")
print(data['Estimated Unemployment Rate (%)'].describe())




# Plot unemployment rate over time
plt.figure(figsize=(12,6))
sns.lineplot(x='Date', y='Estimated Unemployment Rate (%)', data=data)

plt.title("Unemployment Rate Over Time")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Filter data for 2019 onwards to observe COVID impact
covid_data = data[data['Date'].dt.year >= 2019]

plt.figure(figsize=(12,6))
sns.lineplot(x='Date', y='Estimated Unemployment Rate (%)', data=covid_data)

plt.title("Unemployment Rate During COVID Period (2019 onwards)")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Average unemployment by region
region_avg = data.groupby('Region')['Estimated Unemployment Rate (%)'].mean().sort_values(ascending=False)

plt.figure(figsize=(10,8))
sns.barplot(x=region_avg.values[:10], y=region_avg.index[:10])

plt.title("Top 10 Regions by Average Unemployment Rate")
plt.xlabel("Average Unemployment Rate (%)")
plt.ylabel("Region")

plt.tight_layout()
plt.show()