import pandas as pd
import streamlit as st

from inference import REQUIRED_FEATURE_COLUMNS, predict_churn, predict_single_customer


st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


MODEL_OPTIONS = {
    "Logistic Regression — L1": "logistic_regression_l1",
    "Logistic Regression — L2": "logistic_regression_l2",
    "Pruned Decision Tree": "pruned_decision_tree",
    "Random Forest": "random_forest",
    "XGBoost": "xgboost",
}


def apply_dashboard_styles():
    st.markdown(
        """
        <style>
            .stApp {
                background: #111827;
                color: #e5e7eb;
            }
            [data-testid="stHeader"] {
                background: rgba(17, 24, 39, 0.95);
            }
            [data-testid="stSidebar"] {
                background: #0f172a;
                border-right: 1px solid #273449;
            }
            [data-testid="stSidebar"] * {
                color: #e5e7eb;
            }
            h1, h2, h3, p, label, .stMarkdown {
                color: #e5e7eb;
            }
            .hero {
                background: linear-gradient(135deg, #182237 0%, #1e2b4d 100%);
                border: 1px solid #334b73;
                border-radius: 16px;
                padding: 1.5rem 1.75rem;
                margin-bottom: 1.25rem;
            }
            .hero h1 {
                margin: 0;
                color: #f8fafc;
            }
            .hero p {
                margin: 0.5rem 0 0;
                color: #cbd5e1;
            }
            [data-testid="stForm"] {
                background: #182235;
                border: 1px solid #2c3d59;
                border-radius: 14px;
                padding: 1.25rem;
            }
            [data-testid="stFileUploader"] {
                background: #182235;
                border: 1px solid #2c3d59;
                border-radius: 12px;
                padding: 0.75rem;
            }
            .stButton > button, .stDownloadButton > button,
            [data-testid="stFormSubmitButton"] > button {
                background: #4f46e5;
                color: #ffffff;
                border: 0;
                border-radius: 8px;
                font-weight: 600;
            }
            .stButton > button:hover, .stDownloadButton > button:hover,
            [data-testid="stFormSubmitButton"] > button:hover {
                background: #6366f1;
                color: #ffffff;
            }
            .result-card {
                border-radius: 14px;
                padding: 1.25rem 1.5rem;
                margin: 1rem 0;
                border: 1px solid;
            }
            .result-card.yes {
                background: #3a1d2c;
                border-color: #d94678;
            }
            .result-card.no {
                background: #123142;
                border-color: #38bdf8;
            }
            .result-card .label {
                color: #cbd5e1;
                font-size: 0.9rem;
            }
            .result-card .value {
                color: #f8fafc;
                font-size: 2rem;
                font-weight: 700;
                margin-top: 0.2rem;
            }
            [data-testid="stMetric"] {
                background: #182235;
                border: 1px solid #2c3d59;
                border-radius: 12px;
                padding: 0.75rem;
            }
            [data-testid="stDataFrame"] {
                border: 1px solid #2c3d59;
                border-radius: 12px;
                overflow: hidden;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def create_template_csv():
    # A header-only template can be reused for any batch size.
    template = pd.DataFrame(columns=["customerID", *REQUIRED_FEATURE_COLUMNS])
    return template.to_csv(index=False).encode("utf-8")


def build_single_customer_frame():
    with st.form("single_customer_form"):
        st.subheader("Customer information")
        customer_col_1, customer_col_2, customer_col_3 = st.columns(3)
        with customer_col_1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox(
                "Senior citizen",
                [0, 1],
                format_func=lambda value: "Yes (1)" if value == 1 else "No (0)",
            )
        with customer_col_2:
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
        with customer_col_3:
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12, step=1)

        st.subheader("Services")
        service_col_1, service_col_2, service_col_3 = st.columns(3)
        with service_col_1:
            phone_service = st.selectbox("Phone service", ["No", "Yes"])
            multiple_lines = st.selectbox(
                "Multiple lines", ["No phone service", "No", "Yes"]
            )
            internet_service = st.selectbox(
                "Internet service", ["DSL", "Fiber optic", "No"]
            )
        with service_col_2:
            online_security = st.selectbox(
                "Online security", ["No", "Yes", "No internet service"]
            )
            online_backup = st.selectbox(
                "Online backup", ["Yes", "No", "No internet service"]
            )
            device_protection = st.selectbox(
                "Device protection", ["No", "Yes", "No internet service"]
            )
        with service_col_3:
            tech_support = st.selectbox(
                "Tech support", ["No", "Yes", "No internet service"]
            )
            streaming_tv = st.selectbox(
                "Streaming TV", ["No", "Yes", "No internet service"]
            )
            streaming_movies = st.selectbox(
                "Streaming movies", ["No", "Yes", "No internet service"]
            )

        st.subheader("Billing")
        billing_col_1, billing_col_2, billing_col_3 = st.columns(3)
        with billing_col_1:
            contract = st.selectbox(
                "Contract", ["Month-to-month", "One year", "Two year"]
            )
            paperless_billing = st.selectbox("Paperless billing", ["Yes", "No"])
        with billing_col_2:
            payment_method = st.selectbox(
                "Payment method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
        with billing_col_3:
            monthly_charges = st.number_input(
                "Monthly charges", min_value=0.0, max_value=200.0, value=70.35, step=0.01
            )
            total_charges = st.number_input(
                "Total charges", min_value=0.0, max_value=10000.0, value=1000.0, step=0.01
            )

        submitted = st.form_submit_button("Predict churn", use_container_width=True)

    if not submitted:
        return None

    return pd.DataFrame(
        [
            {
                "gender": gender,
                "SeniorCitizen": senior_citizen,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless_billing,
                "PaymentMethod": payment_method,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
            }
        ]
    )


def show_single_customer_mode(model_name):
    st.markdown("### Single Customer")
    st.caption("Enter the customer profile and service details to estimate churn risk.")

    customer_data = build_single_customer_frame()
    if customer_data is None:
        return

    try:
        results = predict_single_customer(model_name, customer_data)
    except (TypeError, ValueError, FileNotFoundError) as error:
        st.error(str(error))
        return

    prediction = results.iloc[0]["Prediction"]
    probability = float(results.iloc[0]["Churn Probability"])
    result_class = "yes" if prediction == "Yes" else "no"

    st.markdown(
        f"""
        <div class="result-card {result_class}">
            <div class="label">Churn prediction</div>
            <div class="value">{prediction}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    probability_col, model_col = st.columns(2)
    probability_col.metric("Churn probability", f"{probability:.1%}")
    model_col.metric("Selected model", next(label for label, key in MODEL_OPTIONS.items() if key == model_name))
    risk_split = pd.DataFrame(
        {"Churn risk": [probability], "No churn": [1.0 - probability]},
        index=["Customer"],
    )
    st.bar_chart(
        risk_split,
        horizontal=True,
        stack=True,
        color=["#d94678", "#38bdf8"],
        height=140,
        alt="Customer churn risk split between churn risk and no churn",
    )


def show_batch_upload_mode(model_name):
    st.markdown("### Batch Upload")
    st.caption("Upload a CSV or XLSX file with one customer per row.")
    uploaded_file = st.file_uploader("Customer data file", type=["csv", "xlsx"])

    if uploaded_file is None:
        return

    try:
        if uploaded_file.name.lower().endswith(".csv"):
            customer_data = pd.read_csv(uploaded_file)
        else:
            customer_data = pd.read_excel(uploaded_file)

        st.info(f"Loaded {len(customer_data):,} customer row(s). Validating and predicting...")
        results = predict_churn(model_name, customer_data)
    except (TypeError, ValueError, FileNotFoundError, pd.errors.ParserError) as error:
        st.error(str(error))
        return

    st.success(f"Predictions generated for {len(results):,} customer row(s).")
    prediction_counts = (
        results["Prediction"]
        .value_counts()
        .reindex(["Yes", "No"], fill_value=0)
        .rename_axis("Prediction")
        .reset_index(name="Customers")
    )
    st.bar_chart(
        prediction_counts,
        x="Prediction",
        y="Customers",
        horizontal=True,
        color="#6366f1",
        sort=False,
        alt="Number of customers predicted Yes versus No",
    )
    st.dataframe(results, use_container_width=True, hide_index=True)
    st.download_button(
        "Download prediction results",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="churn_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )


def main():
    apply_dashboard_styles()

    st.markdown(
        """
        <div class="hero">
            <h1>Customer Churn Predictor</h1>
            <p>Evaluate individual customers or customer batches with the selected trained model.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Model selection")
        selected_label = st.selectbox(
            "Prediction model",
            list(MODEL_OPTIONS),
            index=1,
        )
        st.caption("Logistic Regression — L2 is the recommended default model.")
        st.divider()
        st.subheader("Batch template")
        st.caption("Download the schema and add as many customer rows as needed.")
        st.download_button(
            "Download CSV template",
            data=create_template_csv(),
            file_name="customer_churn_template.csv",
            mime="text/csv",
            use_container_width=True,
        )

    model_name = MODEL_OPTIONS[selected_label]
    mode = st.radio(
        "Prediction mode",
        ["Single Customer", "Batch Upload"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if mode == "Single Customer":
        show_single_customer_mode(model_name)
    else:
        show_batch_upload_mode(model_name)


if __name__ == "__main__":
    main()
