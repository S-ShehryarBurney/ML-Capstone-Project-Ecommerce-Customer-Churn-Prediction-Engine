# E-commerce Customer Churn Prediction Engine

An end-to-end machine-learning capstone project for estimating customer churn from account, service, and billing information. It compares five classification models, packages trained preprocessing-and-model pipelines for reuse, and provides a Streamlit interface for individual and batch predictions.

## Project overview

The engine predicts whether a customer will churn (`Yes` or `No`) and returns the predicted probability of churn. Users can choose among five saved models and either enter one customer in the app or upload a customer file. Batch predictions can be reviewed in the interface and downloaded as CSV.

This project is an analytical prediction tool; it does not automatically contact customers or trigger retention actions.

## Dataset

The project uses the included `data/Telco-Customer-Churn.csv` file:

- 7,043 customer records and 21 columns.
- 19 customer, service, and billing features, plus `customerID` and the `Churn` target.
- Target counts: 5,174 `No` and 1,869 `Yes`.
- `TotalCharges` contains blank strings for some records; these are converted to zero, consistent with the zero-tenure records in the dataset.

The dataset contains an imbalanced target, so the evaluation split is stratified to preserve the churn-class proportions.

## Preprocessing and leakage controls

The shared preprocessing code in `preprocessing.py`:

1. Converts blank `TotalCharges` values to zero and casts the column to numeric.
2. Treats `SeniorCitizen` as categorical.
3. Separates `Churn` as the target and excludes both `Churn` and `customerID` from model features.
4. Uses an 80/20 train/test split with stratification and `random_state=42` for the evaluation workflow.
5. Fits imputers, scaling, and categorical encoding on the training partition only, then applies those fitted transformations to the test partition.

Numerical features use mean imputation and standard scaling. Categorical features use most-frequent imputation and one-hot encoding with unknown categories ignored. The deployment pipelines bundle the same feature cleaning and transformations with each classifier so inference receives raw feature values.

**Evaluation caveat:** The model comparison code uses test accuracy while selecting tree depth/pruning and Random Forest/XGBoost settings. The holdout set therefore participates in model selection; the reported results are useful for project comparison but should not be interpreted as an untouched final benchmark. A separate validation set or nested cross-validation would be appropriate for an unbiased final assessment.

## Models and evaluation

The project implements and deploys the following classifiers:

- Logistic Regression — L1
- Logistic Regression — L2
- Pruned Decision Tree
- Random Forest
- XGBoost

The table below is the final comparison printed by `main.py`, reproduced with the repository's dataset and current evaluation code. Values are percentages; `Yes` (churn) is the positive class for precision, recall, and F1-score.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression — L1 | 80.20% | 64.89% | 55.35% | 59.74% | **84.26%** |
| Logistic Regression — L2 | **80.55%** | 65.72% | **55.88%** | **60.40%** | 84.20% |
| Pruned Decision Tree | 79.42% | 65.44% | 47.59% | 55.11% | 82.13% |
| Random Forest | 78.35% | 61.69% | 48.66% | 54.41% | 81.71% |
| XGBoost | 80.13% | **66.55%** | 50.53% | 57.45% | 84.20% |

Logistic Regression — L2 is the strongest all-round model at the evaluated classification threshold: it has the highest accuracy, recall, and F1-score in this comparison. L1 has the highest ROC-AUC by a small margin, while XGBoost has the highest precision. The Streamlit app defaults to L2 but allows users to select any of the five models.

## Deployment and inference

`train_deployment_models.py` builds end-to-end scikit-learn pipelines that combine feature cleaning, the fitted `ColumnTransformer` preprocessing, and a classifier. It trains these pipelines on the full dataset for deployment and serializes them with `joblib` into `artifacts/`:

- `logistic_regression_l1.joblib`
- `logistic_regression_l2.joblib`
- `pruned_decision_tree.joblib`
- `random_forest.joblib`
- `xgboost.joblib`

The deployment Random Forest uses 80 trees, and the deployment XGBoost model uses 50 estimators with a 0.05 learning rate. The saved pipelines are separate from the holdout models used to report evaluation metrics.

`inference.py` is the shared prediction interface used by the Streamlit app. It maps stable model names to saved artifacts, caches loaded pipelines, validates input types and columns, selects only required model features, normalizes model labels to `Yes`/`No`, and returns churn probabilities for the positive class. If an input includes `customerID`, the identifier is retained in the returned batch results. `predict_single_customer` additionally enforces that exactly one customer row is provided.

## Streamlit application

Run the app from the project root with:

```bash
streamlit run streamlit_app.py
```

The dark-themed interface supports two modes:

### Single Customer

Enter a customer's demographic, service, tenure, and billing details, select a model, and submit the form. The result shows the churn prediction, churn probability, selected model, and a probability split chart for churn risk versus no churn.

### Batch Upload

Upload a CSV or XLSX file with one customer per row. Each row must include all 19 required feature columns listed below. `customerID` is optional and `Churn` is permitted as an optional input column; neither is sent to the model as a feature. The app displays predicted labels and churn probabilities, summarizes the number of `Yes` and `No` predictions, and offers the results as a downloadable CSV.

The sidebar provides a header-only CSV template containing `customerID` and the required feature columns. Required columns are:

```text
gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService,
MultipleLines, InternetService, OnlineSecurity, OnlineBackup,
DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract,
PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges
```

## Setup

Use a Python environment with the packages listed in `requirements.txt`. The repository has been exercised with Python 3.13.

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The five saved model artifacts are included in `artifacts/`. To regenerate them from the included dataset, run:

```bash
python train_deployment_models.py
```

To run the exploratory model evaluation and print the comparison table:

```bash
python main.py
```

The evaluation script also displays diagnostic plots and writes the final metric comparison to the console.

## Using the application

1. Start the app with `streamlit run streamlit_app.py`.
2. Choose a prediction model in the sidebar.
3. Select **Single Customer** to enter one customer profile, or **Batch Upload** to upload a CSV/XLSX file.
4. Review the prediction and probability output. In batch mode, review the results table and use **Download prediction results** to save a CSV.

## Project structure

```text
.
├── artifacts/                      # Saved deployment pipelines (.joblib)
├── data/
│   └── Telco-Customer-Churn.csv    # Dataset used for training and evaluation
├── models/                         # Model training and evaluation routines
├── visualizations/                 # Generated model diagnostic plots
├── inference.py                    # Artifact loading, validation, and prediction
├── main.py                         # Dataset inspection and holdout evaluation
├── preprocessing.py                # Feature cleaning and shared preprocessing
├── streamlit_app.py                # Streamlit prediction interface
├── train_deployment_models.py      # Full-data pipeline training and serialization
└── requirements.txt                # Python dependencies
```