import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# Configure Streamlit page
st.set_page_config(
    page_title="Bank Marketing Classification",
    page_icon="📊",
    layout="wide"
)

# Application title
st.title("Bank Marketing Classification Dashboard")

st.write(
    "This application evaluates five trained machine learning "
    "classification models using uploaded test data."
)

# CSV file upload section
st.header("Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload your test CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    # Read uploaded CSV file
    test_data = pd.read_csv(uploaded_file)

    st.success("Test dataset uploaded successfully!")

    # Display dataset information
    st.subheader("Uploaded Dataset Preview")

    st.write("Dataset shape:", test_data.shape)

    st.dataframe(test_data.head())

    # Model selection section
    st.header("Select Classification Model")

    model_options = {
        "Logistic Regression": "logistic_regression.joblib",
        "Decision Tree": "decision_tree.joblib",
        "KNN": "knn.joblib",
        "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
        "SVM": "svm.joblib"
    }

    selected_model_name = st.selectbox(
        "Choose a classification model",
        list(model_options.keys())
    )

    
    # Load selected trained model
    model_path = os.path.join(
        "model",
        model_options[selected_model_name]
    )

    if os.path.exists(model_path):

        selected_model = joblib.load(model_path)

        st.success(f"{selected_model_name} loaded successfully!")

    else:
        st.error(f"Model file not found: {model_path}")

    # Separate input features and target
    X_test = test_data.drop(columns=["y"])
    y_test = test_data["y"]

    # Generate predictions
    y_pred = selected_model.predict(X_test)

    # Convert target into binary format for AUC
    y_test_binary = (y_test == "yes").astype(int)

    # Get positive class index
    positive_class_index = list(selected_model.classes_).index("yes")

    # Calculate prediction scores for AUC
    if hasattr(selected_model, "predict_proba"):
        y_score = selected_model.predict_proba(X_test)[:, positive_class_index]

    else:
        y_score = selected_model.decision_function(X_test)

        if positive_class_index == 0:
            y_score = -y_score

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)

    auc = roc_auc_score(y_test_binary, y_score)

    precision = precision_score(
        y_test, y_pred, pos_label="yes", zero_division=0
    )

    recall = recall_score(
        y_test, y_pred, pos_label="yes", zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred, pos_label="yes", zero_division=0
    )

    mcc = matthews_corrcoef(y_test, y_pred)

    # Display evaluation results
    st.header("Model Evaluation Results")

    st.subheader(f"Performance of {selected_model_name}")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", f"{accuracy:.4f}")
    col2.metric("AUC Score", f"{auc:.4f}")
    col3.metric("Precision", f"{precision:.4f}")

    col4, col5, col6 = st.columns(3)

    col4.metric("Recall", f"{recall:.4f}")
    col5.metric("F1 Score", f"{f1:.4f}")
    col6.metric("MCC", f"{mcc:.4f}")


    # Display confusion matrix
    st.header("Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=["no", "yes"]
    )

    cm_df = pd.DataFrame(
        cm,
        index=["Actual No", "Actual Yes"],
        columns=["Predicted No", "Predicted Yes"]
    )

    st.write("Confusion Matrix:")
    st.dataframe(cm_df)


    # Compare performance of all five models
    st.header("Comparison of All Classification Models")

    comparison_results = []

    for model_name, filename in model_options.items():

        model_path = os.path.join("model", filename)

        comparison_model = joblib.load(model_path)

        comparison_metrics = {}

        comparison_metrics["Model"] = model_name

        comparison_metrics["Accuracy"] = accuracy_score(
            y_test,
            comparison_model.predict(X_test)
        )

        comparison_predictions = comparison_model.predict(X_test)

        y_test_binary = (y_test == "yes").astype(int)

        positive_class_index = list(
            comparison_model.classes_
        ).index("yes")

        if hasattr(comparison_model, "predict_proba"):
            y_score = comparison_model.predict_proba(X_test)[
                :, positive_class_index
            ]
        else:
            y_score = comparison_model.decision_function(X_test)

            if positive_class_index == 0:
                y_score = -y_score

        comparison_metrics["AUC Score"] = roc_auc_score(
            y_test_binary,
            y_score
        )

        comparison_metrics["Precision"] = precision_score(
            y_test,
            comparison_predictions,
            pos_label="yes",
            zero_division=0
        )

        comparison_metrics["Recall"] = recall_score(
            y_test,
            comparison_predictions,
            pos_label="yes",
            zero_division=0
        )

        comparison_metrics["F1 Score"] = f1_score(
            y_test,
            comparison_predictions,
            pos_label="yes",
            zero_division=0
        )

        comparison_metrics["MCC"] = matthews_corrcoef(
            y_test,
            comparison_predictions
        )

        comparison_results.append(comparison_metrics)

    comparison_df = pd.DataFrame(comparison_results)

    st.dataframe(
        comparison_df.set_index("Model").round(4),
        width="stretch"
    )