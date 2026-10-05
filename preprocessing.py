import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

def clean_features(df):
    df = df.copy()

    # Replace the blanks with 0 and then convert to numeric
    df["TotalCharges"] = df["TotalCharges"].replace(" ", 0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])
    
    #print("\nTotalCharges Data Type after Cleaning:")
    #print(df["TotalCharges"].dtype)
    
    #print("\nRemaining Blank TotalCharges:")
    #print((df["TotalCharges"] == " ").sum())
    
    #print("\nTotalCharges of Previous Blank Rows:")
    #print(df.loc[df["tenure"] == 0, "TotalCharges"])
        
    # SeniorCitizen treated as categorical instead of numerical
    df["SeniorCitizen"] = df["SeniorCitizen"].astype("str")

    return df

def build_preprocessor(numerical_columns, categorical_columns):
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy = "mean")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy = "most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown = "ignore"))
    ])

    return ColumnTransformer([
        ("num", numerical_pipeline, numerical_columns),
        ("cat", categorical_pipeline, categorical_columns)
    ])

def preprocess_data(df):

    df = clean_features(df)
    
    # Separate target from features
    X = df.drop(["Churn", "customerID"], axis = 1)
    y = df["Churn"]

    # Separating Numerical and Categorical Columns
    numerical_columns = X.select_dtypes(include = ['int64', 'float64']).columns
    categorical_columns = X.select_dtypes(include = 'str').columns

    preprocessor = build_preprocessor(numerical_columns, categorical_columns)

    # train test split, stratify = y because the target is imbalanced
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.2, random_state = 42, stratify = y
    )

    # fit preprocessor on training data and transform it
    X_train_processed = preprocessor.fit_transform(X_train)

    # transform test data using preprocessing rules learned from training data
    X_test_processed = preprocessor.transform(X_test)

    return (
        X_train_processed, X_test_processed, y_train, y_test
    )
