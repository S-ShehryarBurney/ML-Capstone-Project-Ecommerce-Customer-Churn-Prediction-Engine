from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from preprocessing import build_preprocessor, clean_features

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "Telco-Customer-Churn.csv"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"

# Selected by the existing cost-complexity pruning experiment.
PRUNED_TREE_CCP_ALPHA = 0.0004954742057680135

# end t0 end deployment pipeline
def build_pipeline(model, numerical_columns, categorical_columns):
    return Pipeline([
        ("feature_cleaning", FunctionTransformer(clean_features, validate = False)),
        ("preprocessor", build_preprocessor(numerical_columns, categorical_columns)),
        ("model", model),
    ])

def create_final_pipelines(numerical_columns, categorical_columns):
    return {
        "logistic_regression_l1": build_pipeline(
            LogisticRegression(
                l1_ratio = 1,
                solver = "liblinear",
                max_iter = 1000,
            ),
            numerical_columns,
            categorical_columns,
        ),
        "logistic_regression_l2": build_pipeline(
            LogisticRegression(
                l1_ratio = 0,
                max_iter = 1000,
            ),
            numerical_columns,
            categorical_columns,
        ),
        "pruned_decision_tree": build_pipeline(
            DecisionTreeClassifier(
                random_state = 42,
                ccp_alpha = PRUNED_TREE_CCP_ALPHA,
            ),
            numerical_columns,
            categorical_columns,
        ),
        "random_forest": build_pipeline(
            RandomForestClassifier(
                n_estimators = 80,
                random_state = 42,
            ),
            numerical_columns,
            categorical_columns,
        ),
        "xgboost": build_pipeline(
            XGBClassifier(
                n_estimators = 50,
                learning_rate = 0.05,
                random_state = 42,
            ),
            numerical_columns,
            categorical_columns,
        ),
    }

def train_and_save_pipelines():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(["customerID", "Churn"], axis = 1)
    y = df["Churn"]

    cleaned_features = clean_features(X)

    numerical_columns = cleaned_features.select_dtypes(include = ["int64", "float64"]).columns.tolist()
    categorical_columns = cleaned_features.select_dtypes(include = "str").columns.tolist()

    pipelines = create_final_pipelines(numerical_columns, categorical_columns)
    ARTIFACTS_DIR.mkdir(exist_ok = True)

    artifact_paths = {}
    for name, pipeline in pipelines.items():
        training_target = y.map({"No": 0, "Yes": 1}) if name == "xgboost" else y
        pipeline.fit(X, training_target)

        artifact_path = ARTIFACTS_DIR / f"{name}.joblib"
        joblib.dump(pipeline, artifact_path)
        artifact_paths[name] = artifact_path
        print(f"Saved {name}: {artifact_path}")

    return artifact_paths, df.drop(["customerID", "Churn"], axis = 1).head(1)

def verify_saved_pipelines(artifact_paths, raw_customer_features):
    for name, artifact_path in artifact_paths.items():
        pipeline = joblib.load(artifact_path)
        prediction = pipeline.predict(raw_customer_features)
        print(f"Verified {name}: prediction={prediction[0]}")

def main():
    artifact_paths, raw_customer_features = train_and_save_pipelines()
    verify_saved_pipelines(artifact_paths, raw_customer_features)

if __name__ == "__main__":
    main()
