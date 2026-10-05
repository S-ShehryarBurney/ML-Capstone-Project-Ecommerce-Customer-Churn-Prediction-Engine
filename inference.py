from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd


# Resolve saved artifacts relative to the project directory.
PROJECT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"

# Frontends select a saved pipeline through these stable model names.
MODEL_ARTIFACTS = {
    "logistic_regression_l1": "logistic_regression_l1.joblib",
    "logistic_regression_l2": "logistic_regression_l2.joblib",
    "pruned_decision_tree": "pruned_decision_tree.joblib",
    "random_forest": "random_forest.joblib",
    "xgboost": "xgboost.joblib",
}

# Model features expected in every prediction request.
REQUIRED_FEATURE_COLUMNS = (
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
)

# Optional metadata is allowed but never sent to the model.
OPTIONAL_COLUMNS = ("customerID", "Churn")
ALLOWED_COLUMNS = set(REQUIRED_FEATURE_COLUMNS + OPTIONAL_COLUMNS)


def available_models():
    """Return the supported model-selection names."""
    return tuple(MODEL_ARTIFACTS)


# Cache loaded artifacts for repeated prediction requests.
@lru_cache(maxsize=len(MODEL_ARTIFACTS))
def load_pipeline(model_name):
    """Load and cache a saved deployment pipeline by its selection name."""
    if model_name not in MODEL_ARTIFACTS:
        supported_models = ", ".join(available_models())
        raise ValueError(
            f"Unknown model '{model_name}'. Choose one of: {supported_models}."
        )

    artifact_path = ARTIFACTS_DIR / MODEL_ARTIFACTS[model_name]
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Saved model artifact not found: {artifact_path}")

    return joblib.load(artifact_path)


def validate_input_data(customer_data):
    """Validate the raw customer DataFrame before model prediction."""
    # Validate the input schema before any pipeline call.
    if not isinstance(customer_data, pd.DataFrame):
        raise TypeError("Customer data must be provided as a pandas DataFrame.")

    if customer_data.empty:
        raise ValueError("Customer data must contain at least one row.")

    duplicate_columns = customer_data.columns[customer_data.columns.duplicated()].tolist()
    if duplicate_columns:
        raise ValueError(f"Duplicate columns are not allowed: {duplicate_columns}.")

    missing_columns = [
        column for column in REQUIRED_FEATURE_COLUMNS if column not in customer_data.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Missing required feature columns: {', '.join(missing_columns)}."
        )

    unexpected_columns = [
        column for column in customer_data.columns if column not in ALLOWED_COLUMNS
    ]
    if unexpected_columns:
        raise ValueError(
            "Unexpected columns are not allowed: "
            f"{', '.join(unexpected_columns)}. "
            "Only customerID and Churn are allowed in addition to the required features."
        )


def _normalize_label(label):
    # Normalize string and numeric model outputs to one label convention.
    if label in ("Yes", 1, "1"):
        return "Yes"
    if label in ("No", 0, "0"):
        return "No"
    raise ValueError(f"Model returned an unsupported churn label: {label!r}.")


def _positive_class_index(pipeline):
    # Find the probability column representing churn = Yes.
    for index, label in enumerate(pipeline.classes_):
        if _normalize_label(label) == "Yes":
            return index
    raise ValueError("The saved model does not contain a positive churn class.")


def predict_churn(model_name, customer_data):
    """Predict churn labels and probabilities for one or more raw customer rows."""
    validate_input_data(customer_data)

    # Pass only raw model features to the saved end-to-end pipeline.
    pipeline = load_pipeline(model_name)
    model_features = customer_data.loc[:, REQUIRED_FEATURE_COLUMNS]

    # Normalize labels and obtain the positive-class probability.
    predictions = [_normalize_label(label) for label in pipeline.predict(model_features)]
    positive_class_index = _positive_class_index(pipeline)
    probabilities = pipeline.predict_proba(model_features)[:, positive_class_index]

    results = pd.DataFrame(
        {
            "Prediction": predictions,
            "Churn Probability": probabilities,
        },
        index=customer_data.index,
    )

    if "customerID" in customer_data.columns:
        # Retain the identifier so batch results can be matched to input rows.
        results.insert(0, "customerID", customer_data["customerID"])

    return results


def predict_single_customer(model_name, customer_data):
    """Predict churn for exactly one raw customer row."""
    # Convenience wrapper that enforces the single-row contract.
    validate_input_data(customer_data)
    if len(customer_data) != 1:
        raise ValueError("Single-customer prediction requires exactly one row.")
    return predict_churn(model_name, customer_data)
