import pandas as pd

from preprocessing import preprocess_data
from models.logistic_regression import run_logistic_regression
from models.decision_tree import run_decision_tree
from models.random_forest import run_random_forest
from models.xgboost import run_xgboost

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

logistic_results = run_logistic_regression(X_train_processed, X_test_processed, y_train, y_test)

print(X_train_processed.shape)
print(X_test_processed.shape)

print(X_train_processed[:5])

decision_tree_results = run_decision_tree(X_train_processed, X_test_processed, y_train, y_test)
random_forest_results = run_random_forest(X_train_processed, X_test_processed, y_train, y_test)
xgboost_results = run_xgboost(X_train_processed, X_test_processed, y_train, y_test)

# table to compare the final performance of all models
final_model_comparison = pd.DataFrame({
    "Metrics": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
    "Logistic L1": [
        logistic_results["L1"]["accuracy"],
        logistic_results["L1"]["precision"],
        logistic_results["L1"]["recall"],
        logistic_results["L1"]["f1"],
        logistic_results["L1"]["roc_auc"]
    ],
    "Logistic L2": [
        logistic_results["L2"]["accuracy"],
        logistic_results["L2"]["precision"],
        logistic_results["L2"]["recall"],
        logistic_results["L2"]["f1"],
        logistic_results["L2"]["roc_auc"]
    ],
    "Pruned Decision Tree": [
        decision_tree_results["pruned"]["accuracy"],
        decision_tree_results["pruned"]["precision"],
        decision_tree_results["pruned"]["recall"],
        decision_tree_results["pruned"]["f1"],
        decision_tree_results["pruned"]["roc_auc"]
    ],
    "Random Forest": [
        random_forest_results["final"]["accuracy"],
        random_forest_results["final"]["precision"],
        random_forest_results["final"]["recall"],
        random_forest_results["final"]["f1"],
        random_forest_results["final"]["roc_auc"]
    ],
    "XGBoost": [
        xgboost_results["final"]["accuracy"],
        xgboost_results["final"]["precision"],
        xgboost_results["final"]["recall"],
        xgboost_results["final"]["f1"],
        xgboost_results["final"]["roc_auc"]
    ]
})

# convert model metric values from decimals to percentages
final_model_comparison.iloc[:, 1:] = final_model_comparison.iloc[:, 1:] * 100

print("\nFinal Models Comparison:")
print(final_model_comparison.to_string(index = False, float_format = "{:.2f}".format))
