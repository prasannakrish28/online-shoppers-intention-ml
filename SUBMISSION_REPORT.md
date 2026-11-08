# BITS Pilani - Work Integrated Learning Programmes Division
## M.Tech (Data Science and Engineering)
### Machine Learning - Assignment 2

**Submission Date**: November 2026  
**Evaluation Marks**: 15 Marks  
**Student Name**: Prasanna Krishnan  
**BITS Student ID**: 2024HT01001  
**Email ID**: prasannakrish328@gmail.com  

---

# 1. Submission Links

### GitHub Repository
- **URL**: [https://github.com/prasannakrish28/online-shoppers-intention-ml](https://github.com/prasannakrish28/online-shoppers-intention-ml)  
Contains complete source code, preprocessing pipeline, model artifacts (`.joblib`), `README.md`, `requirements.txt`, test dataset (`test_data.csv`), and Jupyter notebook.

### Live Streamlit Application
- **URL**: [https://online-shoppers-intention-ml.streamlit.app](https://online-shoppers-intention-ml.streamlit.app)  
Interactive Streamlit application deployed on Streamlit Community Cloud supporting custom CSV upload, model selection, live metrics display, confusion matrix heatmap, and prediction inspection.

---

# 2. Proof of Execution on BITS Virtual Lab

The training script and models were executed and verified in the BITS Virtual Lab environment:

![BITS Virtual Lab Execution](vlab_screenshot.png)

---

# 3. Model Benchmark & Evaluation Report

## a. Problem Statement

In online shopping platforms, predicting whether a website visitor will make a purchase during a browsing session helps businesses with personalized product recommendations, targeted promotions, and customer retention. The goal of this assignment is to train, evaluate, and compare five supervised machine learning classification models to predict whether a visitor session results in a purchase (`Revenue = True / 1`) or exits without purchasing (`Revenue = False / 0`).

The dataset exhibits class imbalance, with approximately 15.5% of sessions resulting in a purchase and 84.5% ending without a purchase. Because simple accuracy can be misleading under severe class imbalance, the models are evaluated across multiple metrics: Accuracy, ROC-AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC). The five classification paradigms implemented and compared are:
1. Logistic Regression
2. Decision Tree Classifier
3. k-Nearest Neighbors (kNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

The models and preprocessing pipelines are persisted using `joblib` and served through an interactive Streamlit web application.

---

## b. Dataset Description

- **Dataset**: Online Shoppers Purchasing Intention Dataset
- **Repository**: UCI Machine Learning Repository (Dataset ID: 468)
- **Reference**: Sakar, C.O., Polat, S.O., Katircioglu, R. et al. *Neural Computing and Applications* 31, 6893-6908 (2019).
- **Instances**: 12,330 sessions
- **Features**: 17 attributes (10 numerical, 7 categorical)
- **Target**: `Revenue` (Binary: `1` = Purchase, `0` = No Purchase)

### Feature Breakdown:
1. `Administrative`: Number of administrative/account management pages visited (Integer).
2. `Administrative_Duration`: Time in seconds spent on administrative pages (Float).
3. `Informational`: Number of informational and policy pages visited (Integer).
4. `Informational_Duration`: Time in seconds spent on informational pages (Float).
5. `ProductRelated`: Number of product and catalog pages visited (Integer).
6. `ProductRelated_Duration`: Time in seconds spent on product pages (Float).
7. `BounceRates`: Percentage of visitors entering the site on that page and exiting without any other interaction (Float).
8. `ExitRates`: Percentage of pageviews for that page that were the last in the session (Float).
9. `PageValues`: Average ecommerce value of pages visited before checkout (Float).
10. `SpecialDay`: Closeness of the browsing date to a retail holiday or festival (Float).
11. `Month`: Month of the year (Categorical: Feb, Mar, May, June, Jul, Aug, Sep, Oct, Nov, Dec).
12. `OperatingSystems`: Operating system identifier code (Categorical/Integer).
13. `Browser`: Web browser identifier code (Categorical/Integer).
14. `Region`: Geographic region identifier code (Categorical/Integer).
15. `TrafficType`: Web traffic acquisition channel (Categorical/Integer).
16. `VisitorType`: Visitor category (`Returning_Visitor`, `New_Visitor`, `Other`).
17. `Weekend`: Boolean flag indicating weekend session (`True` / `False`).
18. `Revenue` (Target): Transaction completion indicator (`1` = Purchase, `0` = No Purchase).

### Data Preprocessing & Validation:
- **Zero Missing Data**: Verified complete data integrity with no missing values.
- **Stratified Partitioning**: 80% training split (9,864 rows) and 20% test split (2,466 rows) preserving class ratio ($y=1: \sim 15.5\%$).
- **Scikit-Learn ColumnTransformer**:
  - `StandardScaler()` applied to all 10 continuous numerical attributes.
  - `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` applied to all 7 categorical attributes.
- **Leakage Prevention**: All transformations are coupled inside Scikit-Learn `Pipeline` objects fitted strictly on training data.

---

## c. Repository File Structure

```text
online-shoppers-intention-ml/
├── app.py                         # Streamlit interactive application
├── requirements.txt               # Python package dependencies
├── README.md                      # Assignment documentation and benchmark
├── SUBMISSION_REPORT.md           # Formal assignment report
├── test_data.csv                  # Held-out 20% test partition (2,466 rows)
├── test_sample.csv                # Sample test partition (500 rows)
├── train_and_evaluate.py          # Root execution runner script
├── .streamlit/
│   └── config.toml                # Server and layout settings
└── model/
    ├── train_models.py            # Complete training and evaluation script
    ├── train_models.ipynb         # Interactive Jupyter Notebook demonstration
    ├── decision_tree.joblib       # Serialized Decision Tree pipeline
    ├── logistic_regression.joblib # Serialized Logistic Regression pipeline
    ├── knn.joblib                 # Serialized K-Nearest Neighbor pipeline
    ├── naive_bayes.joblib         # Serialized Gaussian Naive Bayes pipeline
    ├── svm.joblib                 # Serialized Support Vector Machine pipeline
    └── metrics_summary.json       # Compiled evaluation metrics across all 5 models
```

---

## d. Models Used & Evaluation Metrics

All 5 classification models were trained on the identical 80% training split (9,864 samples) and evaluated on the identical 20% stratified test split (2,466 samples).

### 5-Model Performance Comparison Table:
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.8410 | 0.8932 | 0.4913 | 0.7435 | 0.5917 | 0.5145 |
| **Decision Tree** | **0.8970** | **0.9080** | **0.7119** | 0.5628 | **0.6287** | **0.5751** |
| **kNN** | 0.8755 | 0.7962 | 0.6829 | 0.3665 | 0.4770 | 0.4394 |
| **Naive Bayes** | 0.2729 | 0.7334 | 0.1726 | **0.9738** | 0.2933 | 0.1289 |
| **Support Vector Machine** | 0.8873 | 0.8824 | 0.7047 | 0.4686 | 0.5629 | 0.5149 |

---

## Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | By training with `class_weight='balanced'`, Logistic Regression handles class imbalance well and achieves a high Recall of 0.7435 (identifying 284 out of 382 buyers in the test split) with an AUC of 0.8932. However, balancing the classes leads to 294 false positives, which lowers Precision to 0.4913 and overall Accuracy to 0.8410. It provides a reliable and interpretable baseline. |
| **Decision Tree** | Regularized with `max_depth=6` and `min_samples_split=10` to prevent overfitting, the Decision Tree achieved the best overall results across the benchmark. It delivered the highest Accuracy (0.8970), highest ROC-AUC (0.9080), highest Precision (0.7119), highest F1 Score (0.6287), and highest MCC (0.5751). The tree splits effectively identify critical thresholds on key features such as `PageValues` and `ProductRelated_Duration`. |
| **kNN** | Distance-weighted kNN ($k=7$) achieved good Accuracy (0.8755) and Precision (0.6829), but struggled on minority class detection with a low Recall of 0.3665 and F1 of 0.4770. In the high-dimensional space created by numerical and one-hot encoded features, nearest neighbors are heavily dominated by the majority negative class, leading the model to predict non-purchase for borderline cases. |
| **Naive Bayes** | Gaussian Naive Bayes achieved the highest Recall of 0.9738 (capturing 372 of 382 buyers), but had very low Precision (0.1726) and low Accuracy (0.2729). This occurs because Naive Bayes assumes conditional independence between features, whereas website browsing metrics have strong correlations (e.g., `BounceRates` and `ExitRates` are correlated, as are page counts and duration). Applying continuous Gaussian distributions across one-hot encoded binary features also distorts probabilities, causing 1,783 false positives. |
| **Support Vector Machine** | Using an RBF kernel with probability calibration, SVM achieved an Accuracy of 0.8873, AUC of 0.8824, and Precision of 0.7047. It is more conservative in predicting purchases than Logistic Regression (Recall of 0.4686), but produces a balanced decision boundary resulting in a strong MCC score of 0.5149. |
| **Overall, Winner for your dataset?** | **Winner: Decision Tree Classifier**.<br><br>The Decision Tree is the best performing model on the Online Shoppers Purchasing Intention dataset, leading in Accuracy (0.8970), ROC-AUC (0.9080), Precision (0.7119), F1 Score (0.6287), and MCC (0.5751). In e-commerce session data, user intent often follows clear threshold patterns (such as viewing high-value pages before checkout), which decision trees capture effectively through hierarchical rules. |

---

# 4. Streamlit Application Architecture & Features

The deployed Streamlit application (`app.py`) provides the following interactive capabilities:
1. **Dataset Upload Option (CSV)**: Sidebar uploader supporting custom test data files. Falls back automatically to `test_data.csv` if no file is provided.
2. **Model Selection Dropdown**: Dropdown to toggle between all 5 trained classification pipelines on the fly.
3. **Display of Evaluation Metrics**: Dynamic KPI cards displaying Accuracy, ROC-AUC, Precision, Recall, F1 Score, and MCC, paired with an adjustable classification threshold slider.
4. **Confusion Matrix & Classification Report**: Seaborn heatmap displaying confusion matrix values alongside precision, recall, and support figures.
5. **Prediction Explorer**: Table displaying sample predictions with confidence scores and option to export predictions as CSV.

---

# 5. Summary of Deliverables

- **Repository**: Public GitHub repository containing complete code, pipeline definitions, and serialized models.
- **Web Application**: Streamlit dashboard deployed and accessible online via Streamlit Community Cloud.
- **Models**: Five trained classification algorithms compared on an identical test split.
- **Evaluation**: Six performance metrics computed and tabulated.
- **Execution Proof**: Verified execution on BITS Virtual Lab.
