import pandas as pd

from preprocessing import preprocess_data
from models.logistic_regression import run_logistic_regression

df = pd.read_csv(
    "data/Telco-Customer-Churn.csv"
)

# Dataset Inspection
print("\nFirst 5 Rows:")
print(df.head())

print("\nLast 5 Rows:")
print(df.tail())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nDataset Information:")
print(df.info())

print("\nNumerical Statistics/Info:")
print(df.describe())

print("\nCategorical Statistics/Info:")
print(df.describe(include = "str"))

print("\nMissing Values:")
print(df.isnull().sum())

print("\nBlank Values:")
print((df == " ").sum())

print("\nChurn Counts:")
print(df["Churn"].value_counts()) # raw count of observations in each category

# % instead of raw counts of customers in each churn class
print("\nChurn Percentages:")
print(df["Churn"].value_counts(normalize = True))

print("\nUnique Values per Column:")
print(df.nunique()) # number of unique values in each column

# TotalCharges Specific Inspection
print("\nData Type TotalCharges:")
print(df["TotalCharges"].dtype)

print("\nBlank Spaces TotalCharges:")
print((df["TotalCharges"] == " ").sum())

# Inspecting customers with missing TotalCharges values
print("\nRows with Blank TotalCharges:")
print(df[df["TotalCharges"] == " "])

# Tenure check for customers with balnk TotalCharges (Relationship Check)
print(df.loc[df["TotalCharges"] == " ", "tenure"].value_counts())

print(df.duplicated().sum())
print(df["customerID"].duplicated().sum())

print("\nCategorical Columns:")
print(df.select_dtypes(include = "str").nunique())

print("\nCategories:")
for column in df.select_dtypes(include = "str").columns:
    print(f"\n{column}:")
    print(df[column].unique())

X_train_processed, X_test_processed, y_train, y_test = preprocess_data(df)

run_logistic_regression(X_train_processed, X_test_processed, y_train, y_test)

print(X_train_processed.shape)
print(X_test_processed.shape)

print(X_train_processed[:5])
