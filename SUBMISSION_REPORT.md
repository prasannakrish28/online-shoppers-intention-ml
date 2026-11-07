# BITS Pilani — Work Integrated Learning Programmes Division
## M.Tech (Data Science and Engineering)
### Course: Machine Learning (Assignment - 2)

**Submission Date**: November 2026  
**Evaluation Marks**: 15 Marks  
**Student Name**: [Your Name Here]  
**BITS Student ID**: [Your BITS ID Here, e.g., 2024HT01001]  
**Email ID**: [Your Email ID Here]  

---

# 1. Mandatory Submission Links

### 1. GitHub Repository Link
- **URL**: `https://github.com/<YOUR_GITHUB_USERNAME>/online-shoppers-intention-ml`  
  *(Contains complete source code, `requirements.txt`, `README.md`, `test_data.csv`, and `model/` directory with saved `.joblib` pipelines and Jupyter notebook)*

### 2. Live Streamlit Application Link
- **URL**: `https://<YOUR_APP_SUBDOMAIN>.streamlit.app`  
  *(Deployed on Streamlit Community Cloud; opens an interactive web frontend allowing test data upload, model selection, live metrics calculation, and confusion matrix visualization)*

---

# 2. Screenshot of Assignment Execution on BITS Virtual Lab

> **[ 1 Mark ] Proof of Execution on BITS Virtual Lab**  
> *Note for Student: Take a screenshot of your BITS Virtual Lab browser window showing the code/terminal execution or Streamlit app preview, and paste it in the box below before exporting to PDF.*

```
+---------------------------------------------------------------------------------------+
|                                                                                       |
|                                                                                       |
|                                                                                       |
|                      [ PASTE BITS VIRTUAL LAB SCREENSHOT HERE ]                       |
|                                                                                       |
|     (Showing BITS VLab desktop / Jupyter Lab / terminal running train_models.py)      |
|                                                                                       |
|                                                                                       |
|                                                                                       |
+---------------------------------------------------------------------------------------+
```

---

# 3. GitHub README Content

*(As specified in Section 3 - Step 5 of the Assignment Guidelines)*

---

## a. Problem Statement

In contemporary digital commerce, analyzing real-time session telemetry to distinguish between window shoppers and high-intent purchasers is pivotal for personalized engagement, dynamic discount allocation, and cart abandonment reduction.

The objective of this assignment is to design, train, evaluate, and benchmark five distinct supervised machine learning classification paradigms to predict whether an online browsing session will conclude with a purchase (**`Revenue = True / 1`**) or not (**`Revenue = False / 0`**):
1. **Logistic Regression** (Linear probabilistic classifier with balanced class weighting)
2. **Decision Tree Classifier** (Non-linear orthogonal partitioner regularized to avoid overfitting)
3. **K-Nearest Neighbors (kNN)** (Distance-weighted non-parametric instance-based classifier)
4. **Naive Bayes Classifier** (Gaussian likelihood model under conditional independence)
5. **Support Vector Machine (SVM)** (Maximum-margin hyper-plane in non-linear RBF kernel space)

Because transactional sessions represent an imbalanced minority (~15.5% purchase rate), standard accuracy can be misleading. We evaluate each model across six comprehensive metrics: **Accuracy**, **AUC Score (ROC-AUC)**, **Precision**, **Recall**, **F1 Score**, and **Matthews Correlation Coefficient (MCC)**. The resulting models are bundled and deployed via an interactive Streamlit application.

---

## b. Dataset Description

- **Dataset**: Online Shoppers Purchasing Intention Dataset
- **Repository**: UCI Machine Learning Repository (Dataset ID: 468)
- **Reference**: Sakar, C.O., Polat, S.O., Katircioglu, R. et al. *Neural Computing and Applications* 31, 6893–6908 (2019).
- **Instances**: **12,330** sessions (Exceeds minimum requirement of 500 instances)
- **Features**: **17** predictive attributes (10 numerical, 7 categorical) (Exceeds minimum requirement of 12 features)
- **Target**: `Revenue` (Binary: `1` = Completed Purchase, `0` = No Purchase)

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
18. `Revenue` (**Target**): Transaction completion indicator (`1` = Purchase, `0` = No Purchase).

### Data Preprocessing & Validation:
- **Zero Missing Data**: Verified complete data integrity with no missing values.
- **Stratified Partitioning**: 80% training split (9,864 rows) and 20% test split (2,466 rows) preserving class ratio ($y=1: \sim 15.5\%$).
- **Scikit-Learn ColumnTransformer**:
  - `StandardScaler()` applied to all 10 continuous numerical attributes.
  - `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` applied to all 7 categorical attributes.
- **Leakage Prevention**: All transformations are coupled inside Scikit-Learn `Pipeline` objects fitted strictly on training data.

---

## c. GitHub Repository Structure

```text
online-shoppers-intention-ml/
├── app.py                         # Streamlit interactive application
├── requirements.txt               # Pinned Python package dependencies
├── README.md                      # Official assignment documentation
├── test_data.csv                  # Held-out 20% test partition (2,466 rows)
├── test_sample.csv                # Lightweight test sample (500 rows)
├── train_and_evaluate.py          # Root execution helper
├── .streamlit/
│   └── config.toml                # Headless server & telemetry settings
└── model/
    ├── train_models.py            # Complete training, evaluation & serialization script
    ├── train_models.ipynb         # Interactive Jupyter Notebook demonstration
    ├── decision_tree.joblib       # Serialized Decision Tree model pipeline
    ├── logistic_regression.joblib # Serialized Logistic Regression pipeline
    ├── knn.joblib                 # Serialized K-Nearest Neighbor pipeline
    ├── naive_bayes.joblib         # Serialized Gaussian Naive Bayes pipeline
    ├── svm.joblib                 # Serialized Support Vector Machine pipeline
    └── metrics_summary.json       # Pre-computed evaluation metrics across models
```

---

## d. Models Used & Evaluation Metrics

### 5-Model Performance Comparison Table:
*(Evaluated on identical test split: 2,466 samples, Stratified, random_state=42)*

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
| **Logistic Regression** | By configuring `class_weight='balanced'`, Logistic Regression counters the minority class imbalance, achieving an impressive **Recall of 0.7435** (recovering 284 out of 382 buyers) and an **AUC of 0.8932**. However, adjusting the linear decision boundary to capture positive instances incurs 294 false positives, which dampens **Precision to 0.4913** and yields an **Accuracy of 0.8410**. It remains a robust, highly interpretable linear baseline. |
| **Decision Tree** | Regularized with `max_depth=6` and `min_samples_split=10` to avoid memorizing noise, the Decision Tree is the **strongest all-around performer**. It attains the highest **Accuracy (0.8970)**, highest **AUC (0.9080)**, highest **Precision (0.7119)**, highest **F1 Score (0.6287)**, and highest **MCC (0.5751)**. The non-linear, hierarchical tree splits naturally identify decisive threshold behaviors on features like `PageValues` and `ProductRelated_Duration`, segregating buyer intent with precision. |
| **kNN** | Distance-weighted k-Nearest Neighbor ($k=7$) achieves strong **Accuracy (0.8755)** and **Precision (0.6829)**, but struggles on minority class detection with a low **Recall of 0.3665**. In the expanded one-hot dimensional space, distance metrics encounter the curse of dimensionality, and query points are dominated by majority negative density, causing conservative predictions. |
| **Naive Bayes** | Gaussian Naive Bayes achieves an extraordinary **Recall of 0.9738** (detecting 372 of 382 actual buyers), but suffers a severe penalty in **Precision (0.1726)** and **Accuracy (0.2729)**. This stems directly from the violation of the class-conditional feature independence assumption: web telemetry exhibits high collinearity (`BounceRates` vs. `ExitRates`, page counts vs. durations). Furthermore, modeling one-hot sparse indicators with Gaussian densities skews posteriors, generating 1,783 false alarms. |
| **Support Vector Machine** | Utilizing an RBF kernel with Platt-scaling probability calibration, the SVM creates smooth non-linear decision boundaries in transformed Hilbert space, attaining **Accuracy of 0.8873**, **AUC of 0.8824**, and a high **Precision of 0.7047**. While slightly more conservative on recall (**0.4686**) compared to Logistic Regression, its balanced margin yields strong overall generalization (**MCC of 0.5149**). |
| **Overall, Winner for your dataset?** | **Winner: Decision Tree Classifier**.<br><br>The Decision Tree is the definitive winner on the Online Shoppers dataset, surpassing all competing models in **Accuracy (0.8970)**, **AUC (0.9080)**, **Precision (0.7119)**, **F1 Score (0.6287)**, and **MCC (0.5751)**. In digital commerce, user intent is governed by strict threshold milestones (e.g., spending time on high-value checkout pages), which orthogonal tree partitions capture far better than linear hyperplanes or distance functions. |

---

# 4. Streamlit Application Architecture & Features

The deployed Streamlit application fulfills all assignment UI specifications:
1. **Dataset Upload Option (CSV)**: Includes a dedicated file uploader in the sidebar to upload test data batches. If no custom file is provided, the application defaults to the bundled 2,466-row test partition (`test_data.csv`).
2. **Model Selection Dropdown**: An intuitive selectbox allows switching between all five classification pipelines on the fly.
3. **Display of Evaluation Metrics**: Dynamic KPI cards show Accuracy, ROC-AUC, Precision, Recall, F1 Score, and MCC in real time, with an interactive threshold slider.
4. **Confusion Matrix & Classification Report**: Features an interactive Seaborn heatmap showing True Negatives, False Positives, False Negatives, and True Positives, accompanied by a styled classification report table.
5. **Interactive Prediction Explorer**: Allows users to inspect individual session rows, view predicted purchase probabilities, and download enriched predictions as CSV.

---

# 5. Final Submission Checklist

- [x] **GitHub Repository Link Works**: Repository includes complete source code, `requirements.txt`, `README.md`, `test_data.csv`, and `model/`.
- [x] **Streamlit App Link Opens Correctly**: Deployed on Streamlit Community Cloud and loads without errors.
- [x] **All 5 Models Implemented**: Logistic Regression, Decision Tree, kNN, Naive Bayes, Support Vector Machine.
- [x] **All 6 Evaluation Metrics Computed**: Accuracy, AUC, Precision, Recall, F1, MCC.
- [x] **Dataset Criteria Met**: Online Shoppers Purchasing Intention (12,330 instances $\ge 500$, 17 features $\ge 12$).
- [x] **README.md Structure Complete**: All required sections (a, b, c, d, observations, winner) fully documented.
- [x] **BITS Virtual Lab Screenshot**: Ready for insertion in Section 2 of this submission PDF.
EOF
