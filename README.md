# Spam Email Classification Using Machine Learning

## 1. Problem Statement

Spam email classification is an important application of machine learning that helps distinguish unwanted or fraudulent emails from legitimate messages.

The objective of this project is to develop a machine learning-based spam email classification system that identifies whether an email is spam or not spam using numerical features extracted from email messages.

Five machine learning classification algorithms are implemented and evaluated on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Naive Bayes Classifier
5. Support Vector Machine (SVM)

The performance of each model is evaluated using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

An interactive web application is developed using Streamlit to allow users to upload test data, select a trained model, and view its evaluation results.

## 2. Dataset Description

**Dataset Name:** Spambase Dataset

**Dataset Source:** UCI Machine Learning Repository

The Spambase dataset is used to develop and evaluate the spam email classification models.

| Dataset Property | Description |
|---|---|
| Dataset Name | Spambase |
| Total Instances | 4,601 |
| Number of Features | 57 |
| Target Variable | is_spam |
| Classification Type | Binary Classification |
| Class 0 | Not Spam |
| Class 1 | Spam |
| Feature Type | Numerical |

The dataset contains 57 numerical features representing characteristics of email messages. The target variable `is_spam` indicates whether an email is spam or legitimate.

The dataset is used to train and evaluate five different machine learning classification algorithms.

## 3. GitHub Repository Link

**GitHub Repository:**

https://github.com/varshithajgwda/spam-email-classifier

The repository contains the application source code, dataset, test data, saved machine learning models, evaluation metrics, and dependency requirements.

**Live Streamlit Application:**

https://varshithajgwda-spam-email-classifier-app-ofaxnb.streamlit.app/

## 4. Models Used and Performance Comparison

Five machine learning classification algorithms are implemented using the Spambase dataset.

The following evaluation metrics are used to compare their performance:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

### Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9097 | 0.9627 | 0.9088 | 0.8601 | 0.8838 | 0.8109 |
| Decision Tree | 0.9062 | 0.9335 | 0.9003 | 0.8601 | 0.8798 | 0.8035 |
| KNN | 0.8955 | 0.9528 | 0.9079 | 0.8214 | 0.8625 | 0.7811 |
| Naive Bayes | 0.8076 | 0.9405 | 0.6875 | 0.9494 | 0.7975 | 0.6526 |
| Support Vector Machine | 0.9264 | 0.9737 | 0.9281 | 0.8839 | 0.9055 | 0.8459 |

### Observations on Model Performance

| ML Model Name | Observation About Model Performance |
|---|---|
| Logistic Regression | Achieved an accuracy of 90.97% and an AUC score of 0.9627. It demonstrated good classification performance with balanced precision and recall. |
| Decision Tree | Achieved an accuracy of 90.62%. It provided good classification performance, with results close to Logistic Regression. |
| KNN | Achieved an accuracy of 89.55% and precision of 0.9079. Its recall was comparatively lower at 0.8214. |
| Naive Bayes | Achieved the highest recall of 0.9494, indicating good identification of spam emails. However, its precision was comparatively lower at 0.6875. |
| Support Vector Machine | Achieved the highest accuracy (0.9264), AUC (0.9737), precision (0.9281), F1 score (0.9055), and MCC (0.8459) among all five models. |

### Overall Winner for the Dataset

**Support Vector Machine (SVM)** demonstrated the best overall performance among the five classification algorithms.

It achieved the highest accuracy, AUC, precision, F1 score, and MCC, making it the best-performing model based on the reported evaluation metrics.

## 5. Streamlit Web Application

An interactive web application named **Spam Mail Detector** is developed using Streamlit.

The application allows users to evaluate the performance of saved machine learning models on uploaded test email data.

### Application Features

1. **Test Dataset Upload:** Allows users to upload a CSV file containing 57 numerical features and the `is_spam` target column.
2. **Model Selection:** Provides a dropdown menu to select one of the five trained classification models.
3. **Evaluation Metrics:** Displays Accuracy, AUC, Precision, Recall, F1 Score, and MCC.
4. **Confusion Matrix:** Displays a confusion matrix to visualize the classification results.
5. **Classification Report:** Displays the precision and recall of spam and non-spam classes.

The application loads the previously saved trained models and evaluates them on the uploaded test dataset without retraining.

### Live Application Link

https://varshithajgwda-spam-email-classifier-app-ofaxnb.streamlit.app/

## 6. Project Repository Structure

```text
spam-email-classifier/
│
├── data/
│   ├── spambase.csv
│   └── .gitkeep
│
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   └── svm.joblib
│
├── results/
│   └── metrics.csv
│
├── app.py
├── requirements.txt
├── test_data.csv
└── README.md
```

## 7. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn
- Joblib
- GitHub

## 8. Conclusion

This project demonstrates the implementation and comparison of five machine learning classification algorithms for spam email detection using the Spambase dataset.

The performance evaluation shows that Support Vector Machine achieved the best overall results among the five models.

The Streamlit application provides an interactive interface for uploading test data, selecting trained models, and visualizing their classification performance.

The project demonstrates an end-to-end machine learning workflow involving dataset handling, model evaluation, model saving, GitHub repository management, and web application deployment.
