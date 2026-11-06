from ucimlrepo import fetch_ucirepo

# ---------------------------------------------------------
# 1. Load the UCI Bank Marketing dataset
# ---------------------------------------------------------

dataset = fetch_ucirepo(id=222)

X = dataset.data.features
y = dataset.data.targets

print("Dataset downloaded successfully.")
print("Features shape:", X.shape)
print("Target shape:", y.shape)


# ---------------------------------------------------------
# 2. Define target and features
# ---------------------------------------------------------

target_column = "y"

# We exclude duration because it represents information
# obtained during the last customer contact.
excluded_columns = ["duration"]

# Numerical features used for modelling
numerical_features = [
    "age",
    "balance",
    "day_of_week",
    "campaign",
    "pdays",
    "previous"
]

# Categorical features used for modelling
categorical_features = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome"
]


# ---------------------------------------------------------
# 3. Display our modelling feature groups
# ---------------------------------------------------------

print("\nExcluded features:")
print(excluded_columns)

print("\nNumerical features:")
print(numerical_features)

print("\nCategorical features:")
print(categorical_features)

print("\nTotal modelling features:",
      len(numerical_features) + len(categorical_features))

print("\nTarget:")
print(target_column)

# ---------------------------------------------------------
# 4. Create modelling dataset
# ---------------------------------------------------------

# Select only the features that will be used for modelling
selected_features = numerical_features + categorical_features

X = X[selected_features]

# Convert target from DataFrame to 1-dimensional Series
y = y[target_column]


# ---------------------------------------------------------
# 5. Train-Test Split
# ---------------------------------------------------------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ---------------------------------------------------------
# 6. Display split information
# ---------------------------------------------------------

print("\nModelling dataset shape:")
print("X:", X.shape)
print("y:", y.shape)

print("\nTraining set:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting set:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())

# ---------------------------------------------------------
# 7. Build preprocessing pipelines
# ---------------------------------------------------------

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Numerical preprocessing
numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


# Categorical preprocessing
categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)


# Combine numerical and categorical preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ],
    sparse_threshold=0
)


print("\nPreprocessing pipeline created successfully.")
print("Numerical preprocessing: Median Imputation -> StandardScaler")
print("Categorical preprocessing: Most Frequent Imputation -> One-Hot Encoding")


# ---------------------------------------------------------
# 8. Apply preprocessing transformation
# ---------------------------------------------------------

# Fit on training data and transform training data
X_train_processed = preprocessor.fit_transform(X_train)

# Transform testing data using the fitted preprocessor
X_test_processed = preprocessor.transform(X_test)


# ---------------------------------------------------------
# 9. Display transformation results
# ---------------------------------------------------------

print("\nPreprocessing transformation completed.")

print("Original training shape:", X_train.shape)
print("Processed training shape:", X_train_processed.shape)

print("\nOriginal testing shape:", X_test.shape)
print("Processed testing shape:", X_test_processed.shape)


# ---------------------------------------------------------
# 10. Logistic Regression Classifier
# ---------------------------------------------------------

from sklearn.linear_model import LogisticRegression

# Create Logistic Regression pipeline
logistic_regression_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)


# Train the model
logistic_regression_model.fit(X_train, y_train)


# Generate predictions
y_pred_lr = logistic_regression_model.predict(X_test)


# Display results
print("\nLogistic Regression training completed successfully.")

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

print("\nFirst 10 actual values:")
print(y_test.iloc[:10].values)

print("\nFirst 10 predicted values:")
print(y_pred_lr[:10])

# ---------------------------------------------------------
# 11. Decision Tree Classifier
# ---------------------------------------------------------

from sklearn.tree import DecisionTreeClassifier

# Create Decision Tree pipeline
decision_tree_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(random_state=42))
    ]
)


# Train the model
decision_tree_model.fit(X_train, y_train)


# Generate predictions
y_pred_dt = decision_tree_model.predict(X_test)


# Display results
print("\nDecision Tree training completed successfully.")

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

print("\nFirst 10 actual values:")
print(y_test.iloc[:10].values)

print("\nFirst 10 predicted values:")
print(y_pred_dt[:10])


from sklearn.neighbors import KNeighborsClassifier

knn_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", KNeighborsClassifier(n_neighbors=5))
    ]
)

# Train the KNN model
knn_model.fit(X_train, y_train)

# Make predictions
y_pred_knn = knn_model.predict(X_test)

print("\nKNN training completed successfully.")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

print("\nFirst 10 actual values:")
print(y_test.iloc[:10].values)

print("\nFirst 10 predicted values:")
print(y_pred_knn[:10])


from sklearn.naive_bayes import GaussianNB

naive_bayes_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", GaussianNB())
    ]
)

# Train the Naive Bayes model
naive_bayes_model.fit(X_train, y_train)

# Make predictions
y_pred_nb = naive_bayes_model.predict(X_test)

print("\nGaussian Naive Bayes training completed successfully.")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

print("\nFirst 10 actual values:")
print(y_test.iloc[:10].values)

print("\nFirst 10 predicted values:")
print(y_pred_nb[:10])


from sklearn.svm import SVC

svm_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", SVC(kernel="linear", random_state=42))
    ]
)

# Train the SVM model
svm_model.fit(X_train, y_train)

# Make predictions
y_pred_svm = svm_model.predict(X_test)

print("\nSVM training completed successfully.")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

print("\nFirst 10 actual values:")
print(y_test.iloc[:10].values)

print("\nFirst 10 predicted values:")
print(y_pred_svm[:10])


models = {
    "Logistic Regression": logistic_regression_model,
    "Decision Tree": decision_tree_model,
    "KNN": knn_model,
    "Gaussian Naive Bayes": naive_bayes_model,
    "SVM": svm_model
}

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)
import pandas as pd

def evaluate_model(model, X_test, y_test):

    # Generate predictions
    y_pred = model.predict(X_test)

    # Convert target into binary format
    y_test_binary = (y_test == "yes").astype(int)

    # Get the positive class index
    positive_class_index = list(model.classes_).index("yes")

    # Calculate prediction scores for AUC
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, positive_class_index]

    else:
        y_score = model.decision_function(X_test)

        # Ensure scores correspond to the positive class
        if positive_class_index == 0:
            y_score = -y_score

    # Calculate evaluation metrics
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC Score": roc_auc_score(y_test_binary, y_score),
        "Precision": precision_score(
            y_test, y_pred, pos_label="yes", zero_division=0
        ),
        "Recall": recall_score(
            y_test, y_pred, pos_label="yes", zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, y_pred, pos_label="yes", zero_division=0
        ),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

    return metrics

results = {}

for model_name, model in models.items():

    print(f"Evaluating {model_name}...")

    results[model_name] = evaluate_model(
        model,
        X_test,
        y_test
    )

# Convert results into a DataFrame
results_df = pd.DataFrame(results).T

# Display results
print("\nModel Evaluation Results:")
print(results_df.round(4))


import os
import joblib

# Create model directory if it does not exist
os.makedirs("model", exist_ok=True)

# Define model filenames
model_files = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "SVM": "svm.joblib"
}

# Save all trained models
for model_name, model in models.items():

    file_path = os.path.join("model", model_files[model_name])

    joblib.dump(model, file_path)

    print(f"{model_name} saved successfully at {file_path}")



# Verify saved models
print("\nVerifying saved models...")

for model_name, filename in model_files.items():

    file_path = os.path.join("model", filename)

    loaded_model = joblib.load(file_path)

    print(f"{model_name}: Loaded successfully")


# Save test dataset for Streamlit evaluation
test_data = X_test.copy()

# Add actual target values
test_data[target_column] = y_test.values

# Save as CSV
test_data.to_csv("test_data.csv", index=False)

print("\nTest dataset saved successfully.")
print("Test dataset shape:", test_data.shape)
print("Test dataset location: test_data.csv")