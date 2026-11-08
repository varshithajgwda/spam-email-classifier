# Spam Mail Detector: five classifiers with a Streamlit app

Spam e-mail detection with five classification models and a Streamlit app.

- **GitHub repository:** https://github.com/varshithajgwda/spam-email-classifier
- **Live app:** https://varshithajgwda-spam-email-classifier-app-ofaxnb.streamlit.app/

## a. Problem statement

Unwanted e-mail wastes time and can carry scams, but a filter that blocks genuine mail is just as harmful. This project decides whether an e-mail is spam (1) or not spam (0) from 57 numbers that summarise its content: how often certain words and symbols occur, and how much of the text is in capital letters. Five classifiers are trained on the same train/test split, compared on six metrics, saved as pipelines, and used by a Streamlit app that scores an uploaded test file without retraining.

## b. Dataset description

| Item | Value |
|---|---|
| Dataset name and source | Spambase, UCI Machine Learning Repository (created at Hewlett-Packard Labs). https://archive.ics.uci.edu/dataset/94/spambase |
| Type of problem | Binary classification |
| Number of instances | 4,601 in the original file; 4,210 after removing 391 duplicate rows |
| Number of features | 57 numeric features: 48 word-frequency percentages, 6 character-frequency percentages (; ( [ ! $ #) and 3 capital-letter statistics (average run length, longest run, total capitals) |
| Target variable and classes | `is_spam`: 0 = not spam, 1 = spam |
| Missing values / duplicates | No missing values. 391 exact duplicate rows were dropped before splitting, so the same email cannot appear in both train and test. |
| Preprocessing | All features are numeric, so no encoding is needed. `StandardScaler` is part of the pipeline for Logistic Regression, KNN and SVM, so it is fitted on training data only and saved inside the model file. Decision Tree and Naive Bayes use the raw values. |
| Train / test split | 80/20, stratified on the target, `random_state = 2026`: 3,368 training rows and 842 test rows |
| Class distribution (not spam / spam) | Original: 2,788 / 1,813. After de-duplication: 2,531 / 1,679. Train: 2,025 / 1,343. Test: 506 / 336. |

Precision, Recall and F1 are reported for the spam class (the positive class, label 1). AUC is computed from the predicted spam probabilities, and MCC uses all four cells of the confusion matrix, which makes it reliable for the mild class imbalance (about 60% not spam, 40% spam).

## c. GitHub repository link

https://github.com/varshithajgwda/spam-email-classifier

Files in the repository:

- `app.py` : the Streamlit app, which loads the saved models and never retrains
- `requirements.txt` : dependencies with the versions used for training
- `README.md` : this document
- `test_data.csv` : the 842 held-out test emails with the `is_spam` column
- `data/spambase.csv` : the dataset (57 features plus `is_spam`)
- `model/` : the five saved pipelines (`logistic_regression`, `decision_tree`, `knn`, `naive_bayes` and `svm`, each a `.joblib` file)
- `results/metrics.csv` : the metrics table of all five models
- `spam_classifier.ipynb` : the notebook that loads the data, trains and evaluates the models and saves them

To run it locally: `pip install -r requirements.txt`, then `streamlit run app.py`.

## d. Models used

Results on the 842 test emails (the best value in each column is in bold):

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9097 | 0.9627 | 0.9088 | 0.8601 | 0.8838 | 0.8109 |
| Decision Tree | 0.9062 | 0.9335 | 0.9003 | 0.8601 | 0.8798 | 0.8035 |
| KNN | 0.8955 | 0.9528 | 0.9079 | 0.8214 | 0.8625 | 0.7811 |
| Naive Bayes | 0.8076 | 0.9405 | 0.6875 | **0.9494** | 0.7975 | 0.6526 |
| Support Vector Machine | **0.9264** | **0.9737** | **0.9281** | 0.8839 | **0.9055** | **0.8459** |

Settings: Logistic Regression `max_iter=2000`, `C=1`; Decision Tree `max_depth=7`; KNN `k=9`; Gaussian Naive Bayes with default settings; SVM with RBF kernel, `C=3` and `probability=True`. Random seed 2026 wherever a seed is used.

Confusion matrices (rows: actual, columns: predicted):

| ML Model Name | TN | FP | FN | TP |
|---|---|---|---|---|
| Logistic Regression | 477 | 29 | 47 | 289 |
| Decision Tree | 474 | 32 | 47 | 289 |
| KNN | 478 | 28 | 60 | 276 |
| Naive Bayes | 361 | 145 | 17 | 319 |
| Support Vector Machine | 483 | 23 | 39 | 297 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | A strong linear baseline: accuracy 0.9097, AUC 0.9627 and MCC 0.8109, second best on all three. It marks 29 genuine emails as spam and misses 47 spam emails. Once the features are scaled, a straight boundary separates most spam from normal mail, and it trains in a fraction of a second. |
| Decision Tree | Accuracy 0.9062 and recall 0.8601 (the same as Logistic Regression), with 32 false alarms and 47 missed spam emails. Its AUC (0.9335) is clearly lower than Logistic Regression, KNN and SVM because a depth-7 tree gives only coarse probability scores. It needs no scaling and is easy to explain, but it ranks emails less well. |
| KNN | Accuracy 0.8955 and the lowest recall (0.8214): 60 spam emails slip through. Precision is good (0.9079, 28 false alarms). The 57 features are on very different scales, so it only works after standardisation, and the saved model keeps the whole training set, which makes prediction slower than the other models. |
| Naive Bayes | The weakest overall: accuracy 0.8076, precision 0.6875 and MCC 0.6526. It has the highest recall (0.9494, only 17 spam emails missed) but flags 145 genuine emails as spam. Word frequencies are correlated and mostly zero, which breaks the independence and Gaussian assumptions, so it over-predicts spam. |
| Support Vector Machine | The best model: accuracy 0.9264, AUC 0.9737, precision 0.9281, F1 0.9055 and MCC 0.8459. Only 23 genuine emails are flagged and 39 spam emails are missed. The RBF kernel captures non-linear patterns in the word and capital-letter features that the linear model cannot. |
| **Overall winner** | **Support Vector Machine.** It leads on five of six metrics and loses only Recall, to Naive Bayes (0.9494 against 0.8839), but Naive Bayes pays for that with 145 false alarms. For a spam filter, losing a genuine email is costly, so the high precision of SVM matters most. Logistic Regression is a close second and a good fast alternative. |

## e. Streamlit app

| Required feature | Implemented | Description |
|---|---|---|
| Dataset upload option (CSV, test data only) | Yes | Sidebar file uploader for a test CSV, plus a checkbox to use the bundled `test_data.csv`. The file must contain the 57 feature columns and the `is_spam` column; missing columns are reported with an error message. |
| Model selection dropdown | Yes | Dropdown with all five saved models: Logistic Regression, Decision Tree, KNN, Naive Bayes and Support Vector Machine. |
| Display of evaluation metrics | Yes | Six metric cards: Accuracy, AUC, Precision, Recall, F1 Score and MCC, computed on the uploaded data. |
| Confusion matrix or classification report | Yes | Both are shown side by side: a confusion matrix heat map and a per-class classification report. A preview of the uploaded data is available in an expander. |

The uploaded CSV needs the 57 numeric feature columns and the `is_spam` column; extra columns and a different column order are accepted. The test file used is `test_data.csv` in the repository root. The app only loads the saved pipelines from the `model` folder with joblib and calls `predict` and `predict_proba`; it never trains or fits anything on the uploaded data, so the numbers it shows match the table above.

## Technologies used

Python, pandas, NumPy, scikit-learn, Streamlit, Matplotlib, joblib, GitHub.
