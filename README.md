# Bank Marketing Classification Project

## 1. Problem Statement

The objective of this project is to develop and evaluate machine learning classification models to predict whether a customer will subscribe to a term deposit based on their demographic, financial, and marketing-related information.

The project uses the UCI Bank Marketing dataset and treats the problem as a binary classification task, where:

- **Yes:** The customer subscribed to a term deposit.
- **No:** The customer did not subscribe to a term deposit.

Five machine learning classification algorithms are implemented and evaluated:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

The models are evaluated using six performance metrics: Accuracy, AUC Score, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

The project also includes a deployed Streamlit web application that allows users to upload test data, select a trained classification model, and view its evaluation results without retraining the model.

## 2. Dataset Description

### Dataset Source

The dataset used in this project is the **Bank Marketing Dataset**, obtained from the UCI Machine Learning Repository.

- **Dataset Name:** Bank Marketing
- **Source:** UCI Machine Learning Repository
- **Dataset URL:** https://archive.ics.uci.edu/dataset/222/bank+marketing
- **Total Instances:** 45,211
- **Original Input Features:** 16
- **Target Variable:** y
- **Problem Type:** Binary Classification

### Feature Description

The original dataset contains 16 input features describing customer demographics, financial information, and previous marketing campaign interactions.

For this project, the `duration` feature was excluded because it represents the duration of the last contact. Since the objective is to predict subscription before or during customer contact, using this feature could introduce information leakage.

Therefore, **15 input features** were used for model training.

### Target Variable

The target variable `y` indicates whether a customer subscribed to a term deposit.

| Target | Description | Instances |
|---|---|---:|
| no | Customer did not subscribe | 39,922 |
| yes | Customer subscribed | 5,289 |
| **Total** | | **45,211** |

The dataset is imbalanced, with approximately 88.3% negative instances and 11.7% positive instances.

### Data Preprocessing

The following preprocessing techniques were applied:

1. **Missing Value Handling**
   - Numerical features: Missing values were replaced using the median.
   - Categorical features: Missing values were replaced using the most frequent value.

2. **Categorical Encoding**
   - One-Hot Encoding was applied to categorical features.
   - `handle_unknown="ignore"` was used to handle previously unseen categories.

3. **Feature Scaling**
   - StandardScaler was applied to numerical features.

4. **Train-Test Split**
   - Training data: 80%
   - Testing data: 20%
   - Random state: 42
   - Stratified sampling was used to preserve the target class distribution.

5. **Data Leakage Prevention**
   - Preprocessing was fitted only on training data and then applied to testing data.
   - The complete preprocessing and classification pipelines were saved together with the trained models.

## 3. Project Links

### GitHub Repository

The complete source code, trained machine learning models, test dataset, and project dependencies are available in the GitHub repository.

**Repository Link:**  
https://github.com/kishor3868/bank-marketing-classification

### Live Streamlit Application

The project is deployed using Streamlit Community Cloud. Users can upload a test CSV file, select a classification model, and view its evaluation metrics and confusion matrix.

**Live Application Link:**  
https://kishor-bank-marketing-ml.streamlit.app/ 

## 4. Model Comparison

All five classification models were trained and evaluated using the same training and testing datasets.

The following six evaluation metrics were used to compare their performance:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

### Model Evaluation Results

| Model | Accuracy | AUC Score | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8937 | 0.7512 | 0.6714 | 0.1796 | 0.2834 | 0.3100 |
| Decision Tree | 0.8306 | 0.6101 | 0.2950 | 0.3223 | 0.3080 | 0.2120 |
| KNN | 0.8876 | 0.6915 | 0.5568 | 0.1947 | 0.2885 | 0.2826 |
| Gaussian Naive Bayes | 0.8466 | 0.7340 | 0.3689 | 0.4376 | 0.4003 | 0.3146 |
| SVM (Linear) | 0.8932 | 0.7013 | 0.6503 | 0.1881 | 0.2918 | 0.3105 |

### Performance Highlights

| Evaluation Metric | Best Performing Model | Score |
|---|---|---:|
| Accuracy | Logistic Regression | 0.8937 |
| AUC Score | Logistic Regression | 0.7512 |
| Precision | Logistic Regression | 0.6714 |
| Recall | Gaussian Naive Bayes | 0.4376 |
| F1 Score | Gaussian Naive Bayes | 0.4003 |
| MCC | Gaussian Naive Bayes | 0.3146 |


## 5. Model-wise Observations and Overall Winner

### Model-wise Observations

**1. Logistic Regression**

- Achieved the highest Accuracy (0.8937), AUC Score (0.7512), and Precision (0.6714).
- Demonstrated good performance in identifying customers who are likely to subscribe.
- However, its Recall (0.1796) is low, indicating that it misses many actual subscribers.

**2. Decision Tree Classifier**

- Achieved an Accuracy of 0.8306 and AUC Score of 0.6101.
- Its Recall (0.3223) is higher than Logistic Regression, KNN, and SVM.
- However, it achieved the lowest Accuracy, AUC Score, Precision, and MCC among the five models.

**3. K-Nearest Neighbors (KNN)**

- Achieved an Accuracy of 0.8876 and Precision of 0.5568.
- Its performance is relatively close to Logistic Regression in terms of Accuracy.
- However, its Recall (0.1947) and F1 Score (0.2885) indicate limited performance in identifying actual subscribers.

**4. Gaussian Naive Bayes**

- Achieved the highest Recall (0.4376), F1 Score (0.4003), and MCC (0.3146).
- It identified a larger proportion of actual subscribers compared with the other models.
- However, its Precision (0.3689) is lower, meaning it produces more false positive predictions.

**5. Support Vector Machine (SVM)**

- Achieved an Accuracy of 0.8932 and Precision of 0.6503.
- Its overall performance is close to Logistic Regression.
- However, its Recall (0.1881) and F1 Score (0.2918) indicate that it misses many actual subscribers.

### Overall Winner

Based on the evaluation results, **Gaussian Naive Bayes is considered the overall winner** because it achieved the highest F1 Score (0.4003) and Matthews Correlation Coefficient (0.3146).

Although Logistic Regression achieved the highest Accuracy, AUC Score, and Precision, its Recall was relatively low.

Since the dataset is imbalanced, Accuracy alone is not sufficient to determine the best-performing model. Gaussian Naive Bayes provides a better balance between Precision and Recall, making it the preferred model based on F1 Score and MCC.

However, the choice of the best model ultimately depends on the business objective and the relative importance of identifying actual subscribers versus avoiding false positive predictions.

## 6. Project Structure

```
bank-marketing-classification/
│
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── gaussian_naive_bayes.joblib
│   └── svm.joblib
│
├── .gitignore
├── app.py
├── main.py
├── requirements.txt
├── test_data.csv
└── README.md 
```

## 7. How to Run the Project Locally

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Train the Models

```bash
python main.py
```

This downloads the UCI Bank Marketing dataset, performs preprocessing, trains all five classifiers, evaluates their performance, and saves the trained models in the `model/` directory.

### Step 3: Run the Streamlit Application

```bash
streamlit run app.py
```

Upload `test_data.csv`, select a trained model, and view the evaluation metrics, confusion matrix, and model comparison.

The Streamlit application loads the previously saved models and does not retrain them during evaluation.