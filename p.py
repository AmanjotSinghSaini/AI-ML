import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Data Acquisition
data_path = "C:/Users/Amanjot Singh/Downloads/Customer-Churn-Records.csv"
df = pd.read_csv(data_path)

# Convert non-numeric columns to numeric, setting errors='coerce' to replace non-numeric values with NaN
for col in df.select_dtypes(include=['object']).columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 2. Data Import and Export Using Pandas
print("First 5 rows of the dataset:")
print(df.head())
print("\nDataset Information:")
print(df.info())
print("\nStatistical Summary:")
print(df.describe())
print("\nMissing Values in Each Column:")
print(df.isnull().sum())

# 3. Exploratory Data Analysis (EDA)
# Visualizing the distribution of numerical columns
num_cols = df.select_dtypes(include=['float64', 'int64']).columns

# Histograms for numeric data
df[num_cols].hist(bins=15, figsize=(15, 10))
plt.suptitle("Distribution of Numerical Features")
plt.show()

# Remove any columns that became empty after handling missing values or outliers
num_cols = [col for col in num_cols if not df[col].isnull().all()]

# Dynamically set up rows and columns for boxplots
num_plots = len(num_cols)
cols = 4  # Number of columns in the subplot grid
rows = (num_plots // cols) + (num_plots % cols > 0)

# Boxplots to detect outliers
plt.figure(figsize=(15, rows * 3))
for i, col in enumerate(num_cols, 1):
    plt.subplot(rows, cols, i)
    sns.boxplot(df[col])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# 4. Handling Missing Values and Outliers
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype in ['float64', 'int64']:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

print("\nAfter Handling Missing Values:")
print(df.isnull().sum())

# Handling Outliers using IQR method
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

print("\nDataset shape after handling outliers:", df.shape)

# Export cleaned data to a new CSV file
output_path = "cleaned_dataset.csv"
df.to_csv(output_path, index=False)
print(f"Cleaned data has been saved to {output_path}")



