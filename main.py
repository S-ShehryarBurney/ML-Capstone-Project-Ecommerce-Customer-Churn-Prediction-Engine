import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

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
print(df.describe(include = "object"))

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

# Replace the blanks with 0 and then convert to numeric
df["TotalCharges"] = df["TotalCharges"].replace(" ", 0)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])

print("\nTotalCharges Data Type after Cleaning:")
print(df["TotalCharges"].dtype)

print("\nRemaining Blank TotalCharges:")
print((df["TotalCharges"] == " ").sum())

print("\nTotalCharges of Previous Blank Rows:")
print(df.loc[df["tenure"] == 0, "TotalCharges"])

print(df.duplicated().sum())
print(df["customerID"].duplicated().sum())

print("\nCategorical Columns:")
print(df.select_dtypes(include = "str").nunique())

print("\nCategories:")
for column in df.select_dtypes(include = "str").columns:
    print(f"\n{column}:")
    print(df[column].unique())

# SeniorCitizen treated as categorical instead of numerical
df["SeniorCitizen"] = df["SeniorCitizen"].astype("str")

# Separate target from features
X = df.drop(["Churn", "customerID"], axis = 1)
y = df["Churn"]

# Separating Numerical and Categorical Columns
numerical_columns = X.select_dtypes(include = ['int64', 'float64']).columns

for column in numerical_columns:
    print("\nNumerical Column:")
    print(df[column].unique())

categorical_columns = X.select_dtypes(include = 'str').columns

for column in categorical_columns:
    print("\nCategorical Column:")
    print(df[column].unique())

# Pipelines (Numerical and Categorical)
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy = "mean")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy = "most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown = "ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_columns),
    ("cat", categorical_pipeline, categorical_columns)
])

# train test split, stratify = y because the target is imbalanced
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 42, stratify = y
)

# fit preprocessor on training data and transform it
X_train_processed = preprocessor.fit_transform(X_train)

# trasnform test data using preprocessing rules learned from training data
X_test_processed = preprocessor.transform(X_test)

print(X_train_processed.shape)
print(X_test_processed.shape)

feature_names = preprocessor.get_feature_names_out()
print(feature_names)

# Inspecting transformed data
print(X_train_processed.shape)
print(X_train_processed[:5])

model = LogisticRegression(max_iter = 1000)
model.fit(X_train_processed, y_train)

y_pred = model.predict(X_test_processed)
print(y_pred[:20])

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

cm = confusion_matrix(y_test, y_pred)
print("cm:", cm)

recall = recall_score(y_test, y_pred, pos_label = "Yes")
print("Recall Score:", recall)

precision = precision_score(y_test, y_pred, pos_label = "Yes")
print("Precision Score:", precision)

f1 = f1_score(y_test, y_pred, pos_label = "Yes")
print("F1-Score:", f1)

# model's predicted probability of churn = yes
y_prob = model.predict_proba(X_test_processed)[:, 1]
print("Model's Predicted Probability for Churn = Yes:", y_prob)

roc_auc = roc_auc_score(y_test, y_prob)
print("ROC_AUC Score:", roc_auc)

fpr, tpr, thresholds = roc_curve(y_test, y_prob, pos_label = "Yes")
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - LogisticRegression")
plt.show()

# L1 Regularization
L1_model = LogisticRegression(
    l1_ratio = 1,
    solver = "liblinear",
    max_iter = 1000
)

L1_model.fit(X_train_processed, y_train)

L1_y_pred = L1_model.predict(X_test_processed)

L1_accuracy = accuracy_score(y_test, L1_y_pred)
print("L1_accuracy:", L1_accuracy)

L1_cm = confusion_matrix(y_test, L1_y_pred)
print("L1_cm:", L1_cm)

L1_recall = recall_score(y_test, L1_y_pred, pos_label = "Yes")
print("L1 Recall Score:", L1_recall)

L1_precision = precision_score(y_test, L1_y_pred, pos_label = "Yes")
print("L1 Precision Score:", L1_precision)

L1_f1 = f1_score(y_test, L1_y_pred, pos_label = "Yes")
print("L1 F1-Score:", L1_f1)

L1_y_prob = L1_model.predict_proba(X_test_processed)[:, 1]
print("L1 Model's Predicted Probability for Churn = Yes:", L1_y_prob)

L1_roc_auc = roc_auc_score(y_test, L1_y_prob)
print("L1 ROC_AUC Score:", L1_roc_auc)

# L2 Regularization
L2_model = LogisticRegression(
    l1_ratio = 0,
    max_iter = 1000
)

L2_model.fit(X_train_processed, y_train)

L2_y_pred = L2_model.predict(X_test_processed)

L2_accuracy = accuracy_score(y_test, L2_y_pred)
print("L2_accuracy:", L2_accuracy)

L2_cm = confusion_matrix(y_test, L2_y_pred)
print("L2_cm:", L2_cm)

L2_recall = recall_score(y_test, L2_y_pred, pos_label = "Yes")
print("L2 Recall Score:", L2_recall)

L2_precision = precision_score(y_test, L2_y_pred, pos_label = "Yes")
print("L2 Precision Score:", L2_precision)

L2_f1 = f1_score(y_test, L2_y_pred, pos_label = "Yes")
print("L2 F1-Score:", L2_f1)

L2_y_prob = L2_model.predict_proba(X_test_processed)[:, 1]
print("L2 Model's Predicted Probability for Churn = Yes:", L2_y_prob)

L2_roc_auc = roc_auc_score(y_test, L2_y_prob)
print("L2 ROC_AUC Score:", L2_roc_auc)

comparison = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
    "L1": [L1_accuracy, L1_precision, L1_recall, L1_f1, L1_roc_auc],
    "L2": [L2_accuracy, L2_precision, L2_recall, L2_f1, L2_roc_auc]
})

comparison["L1"] = comparison["L1"] * 100
comparison["L2"] = comparison["L2"] * 100

print("\nL1 vs L2 Logistic Regression:")
print(comparison.to_string(index = False, float_format = "%.2f%%"))