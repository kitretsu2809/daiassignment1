import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the dataset (assuming it’s provided as a CSV string or file)
# For this example, I’ll use the provided data directly
data = pd.read_csv("Salaries.csv")  # Replace with actual file path if needed
# Alternatively, if working directly with the provided text, parse it accordingly

# For demonstration, I’ll assume the data is already in a DataFrame
# Here’s how it looks based on the provided document

df = pd.DataFrame(data)
print("Dataset Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())
print("\nFirst Few Rows:")
print(df.head())

# Data Cleaning
# Count missing values
print("\nMissing Values Before Imputation:")
print(df.isnull().sum())

# Impute missing values with the median
for col in ['yrs.since.phd', 'yrs.service', 'salary']:
    df[col] = df[col].fillna(df[col].median())

# Verify no missing values remain
print("\nMissing Values After Imputation:")
print(df.isnull().sum())

# Check for duplicates
print("\nNumber of Duplicate Rows:", df.duplicated().sum())

# Remove duplicates if any
df = df.drop_duplicates()

#Detecting and Treating Outliers

for col in ['yrs.since.phd', 'yrs.service', 'salary']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"Outliers in {col}: {len(outliers)} rows")

#Standardizing Categorical Values

print("\nUnique Values in Categorical Columns:")
print("Rank:", df['rank'].unique())
print("Discipline:", df['discipline'].unique())
print("Sex:", df['sex'].unique())

#Exploratory Data Analysis (EDA)
#Examine summary statistics and distributions.
for col in ['yrs.since.phd', 'yrs.service', 'salary']:
    print(f"\nSummary Statistics for {col}:")
    print(df[col].describe())
    print(f"Skewness: {df[col].skew():.2f}")
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()
#Examine frequency distributions.
for col in ['rank', 'discipline', 'sex']:
    print(f"\nFrequency Distribution for {col}:")
    print(df[col].value_counts(normalize=True))
    plt.figure(figsize=(6, 4))
    sns.countplot(x=col, data=df)
    plt.title(f'Distribution of {col}')
    plt.show()

# Bivariate Analysis

# Explore relationships between pairs of variables.
# Numerical-Numerical Relationships

# Examine correlations and scatter plots.

corr = df[['yrs.since.phd', 'yrs.service', 'salary']].corr()
plt.figure(figsize=(6, 4))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

plt.figure(figsize=(6, 4))
sns.scatterplot(x='yrs.since.phd', y='salary', data=df)
plt.title('Years Since PhD vs. Salary')
plt.show()

# Categorical-Numerical Relationships

# Use box plots to compare distributions.

plt.figure(figsize=(8, 5))
sns.boxplot(x='rank', y='salary', data=df)
plt.title('Salary Distribution by Rank')
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(x='sex', y='salary', data=df)
plt.title('Salary Distribution by Sex')
plt.show()

# Categorical-Categorical Relationships

# Use count plots with hues.

plt.figure(figsize=(8, 5))
sns.countplot(x='rank', hue='sex', data=df)
plt.title('Rank Distribution by Sex')
plt.show()

# Multivariate Analysis

# Examine interactions among multiple variables.
# Pair Plots

# Visualize relationships with a categorical hue.

sns.pairplot(df, hue='rank', diag_kind='kde')
plt.show()

# Grouped Comparisons

# Analyze combined effects using a pivot table.

pivot = df.pivot_table(index='rank', columns='sex', values='salary', aggfunc='mean')
plt.figure(figsize=(6, 4))
sns.heatmap(pivot, annot=True, cmap='YlGnBu', fmt='.0f')
plt.title('Average Salary by Rank and Sex')
plt.show()