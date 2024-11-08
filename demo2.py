#
Here are some Python sample programs tailored for a Data Analyst. 
These examples cover common tasks such as data manipulation, analysis,
visualization, and reporting using popular libraries like Pandas, NumPy, Matplotlib, and Seaborn.

#1. Data Manipulation with Pandas
This script reads a CSV file, cleans the data, and performs some basic analysis.

import pandas as pd

# Load the dataset
df = pd.read_csv('sales_data.csv')

# Display the first few rows
print("First 5 rows of the dataset:")
print(df.head())

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Fill missing values with the mean
df['Sales'] = df['Sales'].fillna(df['Sales'].mean())

# Calculate total sales and average sales
total_sales = df['Sales'].sum()
average_sales = df['Sales'].mean()

print(f"\nTotal Sales: {total_sales}")
print(f"Average Sales: {average_sales}")

#Data Visualization with Matplotlib and Seaborn
This example creates a bar plot and a box plot to analyze sales data.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('sales_data.csv')

# Bar plot of total sales by product
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Product', y='Sales', estimator=sum, ci=None)
plt.title("Total Sales by Product")
plt.xlabel("Product")
plt.ylabel("Total Sales")
plt.show()

# Box plot to show the distribution of sales
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Product', y='Sales')
plt.title("Sales Distribution by Product")
plt.xlabel("Product")
plt.ylabel("Sales")
plt.show()

#3. Correlation Analysis
This script checks the correlation between different numeric columns in a dataset.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('financial_data.csv')

# Calculate the correlation matrix
correlation_matrix = df.corr()

# Display the correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

#Descriptive Statistics
This example calculates key statistics for a dataset, helping to understand its basic properties.

import pandas as pd

# Load the dataset
df = pd.read_csv('customer_data.csv')

# Summary statistics
summary = df.describe()

# Display summary statistics
print("Summary Statistics:")
print(summary)

# Calculate additional statistics
median_income = df['Income'].median()
income_variance = df['Income'].var()

print(f"\nMedian Income: {median_income}")
print(f"Income Variance: {income_variance}")

#5. Time Series Analysis
Analyzes trends over time in a dataset containing date and sales information.

import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('time_series_data.csv', parse_dates=['Date'], index_col='Date')

# Resample the data to get monthly sales
monthly_sales = df['Sales'].resample('M').sum()

# Plot the time series
plt.figure(figsize=(12, 6))
plt.plot(monthly_sales, marker='o', color='b')
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.grid()
plt.show()

#6. Data Cleaning and Transformation
Cleans and transforms a dataset by handling missing values, converting data types, and creating new columns.

import pandas as pd

# Load the dataset
df = pd.read_csv('employee_data.csv')

# Drop duplicate rows
df.drop_duplicates(inplace=True)

# Fill missing values in 'Salary' column with the median
df['Salary'].fillna(df['Salary'].median(), inplace=True)

# Convert 'Hire Date' to datetime format
df['Hire Date'] = pd.to_datetime(df['Hire Date'])

# Create a new column 'Years of Experience'
df['Years of Experience'] = 2024 - df['Hire Date'].dt.year

print("Cleaned Data:")
print(df.head())

#7. Simple Linear Regression Analysis
Performs a linear regression to understand the relationship between advertising spend and sales.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the dataset
df = pd.read_csv('advertising.csv')

# Define the predictor (X) and response (y) variables
X = df[['Advertising Spend']].values
y = df['Sales'].values

# Create the model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Plot the regression line
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.xlabel("Advertising Spend")
plt.ylabel("Sales")
plt.title("Linear Regression Analysis")
plt.legend()
plt.show()

# Display the model coefficients
print(f"Intercept: {model.intercept_}")
print(f"Slope: {model.coef_[0]}")

#8. Automating Data Reports with Python
Generates a summary report and saves it as a CSV file.

import pandas as pd

# Load the dataset
df = pd.read_csv('sales_data.csv')

# Generate a summary report
report = df.groupby('Product')['Sales'].sum().reset_index()

# Save the report as a CSV file
report.to_csv('sales_summary_report.csv', index=False)

print("Report saved as 'sales_summary_report.csv'.")

#These examples cover various aspects of data analysis tasks, 
including data cleaning, exploration, visualization, statistical analysis,
 and automation. Each script can be adapted based on specific datasets and requirements. Let me know if you need more specific examples or have a particular focus in mind!

