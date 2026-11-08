# Online Shoppers Purchasing Intention - Classification Benchmark & Deployment

**Course**: M.Tech (Data Science and Engineering) - Machine Learning  
**Institution**: BITS Pilani (Work Integrated Learning Programmes Division)  
**Assignment**: Assignment - 2  

---

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
- **Source**: UCI Machine Learning Repository (Dataset ID: 468)  
  *Reference*: Sakar, C.O., Polat, S.O., Katircioglu, R. et al. *Neural Comput & Applic* 31, 6893-6908 (2019).  
  *URL*: `https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset`
- **Total Instances**: 12,330 sessions
- **Total Features**: 17 attributes (10 numerical, 7 categorical)
- **Target Variable**: `Revenue` (Boolean: `True` indicating purchase completed, `False` indicating session ended without purchase)

### Feature Inventory:
| # | Feature Name | Data Type | Description |
|---|--------------|-----------|-------------|
| 1 | `Administrative` | Integer (Numerical) | Number of administrative/account management pages visited |
| 2 | `Administrative_Duration` | Float (Numerical) | Total time (in seconds) spent on administrative pages |
| 3 | `Informational` | Integer (Numerical) | Number of informational/help/policy pages visited |
| 4 | `Informational_Duration` | Float (Numerical) | Total time (in seconds) spent on informational pages |
| 5 | `ProductRelated` | Integer (Numerical) | Number of catalog and product-related pages visited |
| 6 | `ProductRelated_Duration` | Float (Numerical) | Total time (in seconds) spent browsing product pages |
| 7 | `BounceRates` | Float (Numerical) | Percentage of visitors entering the site on that page and leaving without triggering further requests |
| 8 | `ExitRates` | Float (Numerical) | Percentage of pageviews that were the final interaction in the session |
| 9 | `PageValues` | Float (Numerical) | Average ecommerce value of pages visited before a transaction |
| 10 | `SpecialDay` | Float (Numerical) | Closeness of browsing date to a specific holiday/e-commerce event (e.g. Valentine's Day, Mother's Day) |
| 11 | `Month` | Categorical (String) | Month of the browsing session (Feb, Mar, May, June, Jul, Aug, Sep, Oct, Nov, Dec) |
| 12 | `OperatingSystems` | Categorical (Integer) | Operating system identifier used by the client device |
| 13 | `Browser` | Categorical (Integer) | Browser identifier used by the visitor |
| 14 | `Region` | Categorical (Integer) | Geographic region identifier from which the session originated |
| 15 | `TrafficType` | Categorical (Integer) | Traffic acquisition channel (e.g., direct, search engine, banner, newsletter) |
| 16 | `VisitorType` | Categorical (String) | Visitor segment: `Returning_Visitor`, `New_Visitor`, `Other` |
| 17 | `Weekend` | Boolean | Binary flag indicating whether the session occurred on a weekend |
| **Target** | `Revenue` | Binary (0 / 1) | Purchase completion indicator: `1` = Purchase, `0` = No Purchase |

### Preprocessing & Data Partitioning:
- **Missing Values**: The dataset contains zero null or missing entries.
- **Train-Test Split**: Stratified 80:20 split (`random_state=42`), producing 9,864 training instances and 2,466 test instances.
- **Feature Scaling**: Numerical attributes are normalized using `StandardScaler` fitted on training data.
- **Categorical Encoding**: Categorical attributes are encoded via `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
- **Pipeline Architecture**: All preprocessors and estimators are encapsulated inside Scikit-Learn `Pipeline` objects to prevent data leakage.

---

## c. Repository and Deployment Links

- **GitHub Repository**: [https://github.com/prasannakrish28/online-shoppers-intention-ml](https://github.com/prasannakrish28/online-shoppers-intention-ml)
- **Live Streamlit Application**: [https://online-shoppers-intention-ml.streamlit.app](https://online-shoppers-intention-ml.streamlit.app)

### Repository File Structure:
```text
online-shoppers-intention-ml/
├── app.py                         # Interactive Streamlit application frontend
├── requirements.txt               # Python dependency specification
├── README.md                      # Project documentation and benchmark report
├── SUBMISSION_REPORT.md           # Assignment submission report
├── test_data.csv                  # Held-out test split (2,466 instances)
├── test_sample.csv                # Sample test partition (500 instances)
├── train_and_evaluate.py          # Root execution helper script
├── .streamlit/
│   └── config.toml                # Streamlit server and theme configuration
└── model/                         # Model artifacts and training code
    ├── train_models.py            # Python script for model training and evaluation
    ├── train_models.ipynb         # Jupyter Notebook version of the pipeline
    ├── logistic_regression.joblib # Serialized Logistic Regression pipeline
    ├── decision_tree.joblib       # Serialized Decision Tree pipeline
    ├── knn.joblib                 # Serialized K-Nearest Neighbor pipeline
    ├── naive_bayes.joblib         # Serialized Gaussian Naive Bayes pipeline
    ├── svm.joblib                 # Serialized Support Vector Machine pipeline
    └── metrics_summary.json       # Compiled evaluation metrics across all 5 models
```

---

## d. Models Used

All 5 classification models were trained on the identical 80% training split (9,864 samples) and evaluated on the identical 20% stratified test split (2,466 samples).

### Comparison Table:
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

## Streamlit Web Application

The interactive web application (`app.py`) provides the following functionality:
1. **Dataset Upload (CSV)**: Users can upload custom test CSV files. Defaults to `test_data.csv` if no custom file is uploaded.
2. **Model Selection**: Switch dynamically between any of the 5 trained classification pipelines.
3. **Evaluation Metrics Display**: Dynamic cards displaying Accuracy, ROC-AUC, Precision, Recall, F1 Score, and MCC, with an interactive threshold slider.
4. **Confusion Matrix & Classification Report**: Visual heatmap of true vs. predicted labels alongside a detailed classification report.
5. **Prediction Inspector**: View test predictions with predicted probabilities and export results to CSV.

---

## Local Setup & Execution

```bash
# 1. Clone repository
git clone https://github.com/prasannakrish28/online-shoppers-intention-ml.git
cd online-shoppers-intention-ml

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train and evaluate models
python model/train_models.py

# 4. Launch Streamlit web application
streamlit run app.py
```
