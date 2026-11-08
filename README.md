# Online Shoppers Purchasing Intention - Classification Benchmark & Deployment

**Course**: M.Tech (Data Science and Engineering) — Machine Learning  
**Institution**: BITS Pilani (Work Integrated Learning Programmes Division)  
**Assignment**: Assignment - 2  
**Semester / Academic Year**: 2026-2027  

---

## a. Problem Statement

In contemporary e-commerce platforms, understanding and anticipating visitor purchase intent in real time is critical for personalized recommendations, promotional coupon targeting, and dynamic customer retention. The objective of this project is to build, evaluate, and benchmark multiple machine learning classification models to predict whether an online browsing session will culminate in a financial transaction (**`Revenue = True / 1`**) or abandon without purchasing (**`Revenue = False / 0`**).

Because purchasing visitors represent a minority class (~15.5%) compared to casual browsing sessions (~84.5%), standard classification accuracy alone is an insufficient measure of efficacy. We systematically train and evaluate 5 core classification paradigms—**Logistic Regression**, **Decision Tree Classifier**, **K-Nearest Neighbors (kNN)**, **Naive Bayes (GaussianNB)**, and **Support Vector Machine (SVM)**—using comprehensive performance metrics (Accuracy, ROC-AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient). Finally, we operationalize the best-performing pipeline through an interactive **Streamlit web application** deployed on Streamlit Community Cloud.

---

## b. Dataset Description

- **Dataset Name**: Online Shoppers Purchasing Intention Dataset
- **Source**: UCI Machine Learning Repository (Dataset ID: 468)  
  *Reference*: Sakar, C.O., Polat, S.O., Katircioglu, R. et al. *Neural Comput & Applic* 31, 6893–6908 (2019).  
  *Direct URL*: `https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset`
- **Total Instances (Rows)**: 12,330 sessions (Exceeds the assignment requirement of $\ge 500$ instances)
- **Total Features (Columns)**: 17 feature attributes + 1 target column (Exceeds the assignment requirement of $\ge 12$ features)
- **Target Variable**: `Revenue` (Boolean: `True` indicating purchase completed, `False` indicating session ended without purchase).

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
- **Missing Values**: The dataset is fully validated with zero null/missing entries.
- **Train-Test Split**: Stratified 80:20 split (`random_state=42`), producing 9,864 training instances and 2,466 test instances.
- **Feature Scaling**: Numerical attributes are normalized using `StandardScaler` fitted strictly on training data.
- **Categorical Encoding**: Categorical attributes are encoded via `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
- **Pipeline Architecture**: All preprocessors and estimators are encapsulated inside Scikit-Learn `Pipeline` objects to prevent data leakage and guarantee atomic inference.

---

## c. GitHub Repository Link

- **GitHub Repository**: `https://github.com/prasannakrish28/online-shoppers-intention-ml`
- **Live Streamlit App**: `[https://<YOUR_APP_SUBDOMAIN>.streamlit.app](https://<YOUR_APP_SUBDOMAIN>.streamlit.app)`

*(Note: Replace `<YOUR_GITHUB_USERNAME>` and `<YOUR_APP_SUBDOMAIN>` with your personal GitHub handle and Streamlit Community Cloud URL upon deployment)*

### Repository File Structure:
```text
project-folder/
├── app.py                     # Interactive Streamlit application frontend
├── requirements.txt           # Python dependency specification for deployment
├── README.md                  # Comprehensive project documentation & benchmark report
├── test_data.csv              # Held-out test split (2,466 instances) used for evaluation
├── test_sample.csv            # Lightweight test sample (500 instances) for rapid testing
├── .streamlit/
│   └── config.toml            # Streamlit server and theme configuration
└── model/                     # Model artifacts and training code
    ├── train_models.py        # Python script for end-to-end model training & evaluation
    ├── train_models.ipynb     # Interactive Jupyter Notebook version of the pipeline
    ├── logistic_regression.joblib  # Serialized Logistic Regression pipeline
    ├── decision_tree.joblib        # Serialized Decision Tree pipeline
    ├── knn.joblib                  # Serialized K-Nearest Neighbor pipeline
    ├── naive_bayes.joblib          # Serialized Gaussian Naive Bayes pipeline
    ├── svm.joblib                  # Serialized Support Vector Machine pipeline
    └── metrics_summary.json        # Compiled evaluation metrics across all 5 models
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
| **Logistic Regression** | By utilizing `class_weight='balanced'`, Logistic Regression proactively compensates for the minority positive class, achieving a high **Recall of 0.7435** (capturing 284 out of 382 actual buyers) and a strong **ROC-AUC of 0.8932**. However, shifting the decision plane to penalize false negatives increases false alarms (294 false positives), capping **Precision at 0.4913** and overall **Accuracy at 0.8410**. It serves as an excellent, interpretable probabilistic baseline. |
| **Decision Tree** | Regularized with `max_depth=6` and `min_samples_split=10` to curb overfitting, the Decision Tree delivers the **best overall performance** across the benchmark. It achieves the highest **Accuracy (0.8970)**, highest **ROC-AUC (0.9080)**, highest **Precision (0.7119)**, highest **F1 Score (0.6287)**, and highest **MCC (0.5751)**. The tree structure naturally isolates critical non-linear split points on features such as `PageValues` and `ProductRelated_Duration`, making it robust against high-dimensional noise. |
| **kNN** | Distance-weighted k-Nearest Neighbor ($k=7$) achieves respectable **Accuracy (0.8755)** and **Precision (0.6829)**, but suffers on minority class coverage with a low **Recall of 0.3665** and **F1 of 0.4770**. In the expanded 17+ dimensional space, Euclidean distance metrics experience metric dispersion (curse of dimensionality), and local neighborhoods are heavily dominated by majority negative samples, biasing predictions toward non-purchase. |
| **Naive Bayes** | Gaussian Naive Bayes achieves the highest **Recall of 0.9738** (detecting 372 of 382 buyers), but suffers a severe drop in **Precision (0.1726)** and **Accuracy (0.2729)**. This is a classic violation of Naive Bayes' core assumption: attributes are assumed to be conditionally independent, whereas web metrics exhibit severe multicollinearity (e.g., `BounceRates` is strongly correlated with `ExitRates`, and page counts correlate with durations). Additionally, fitting continuous Gaussian densities over one-hot sparse indicators distorts posterior probabilities, leading to 1,783 false positives. |
| **Support Vector Machine** | Utilizing an RBF kernel with probability calibration, the SVM establishes an effective non-linear margin in transformed Hilbert space, attaining **Accuracy of 0.8873**, **AUC of 0.8824**, and a high **Precision of 0.7047**. While slightly more conservative in capturing minority positive instances (**Recall of 0.4686**) compared to Logistic Regression, its balanced margin yields strong overall consistency (**MCC of 0.5149**). |
| **Overall, Winner for your dataset?** | **Winner: Decision Tree Classifier**.<br><br>The Decision Tree is the clear champion for the Online Shoppers Purchasing Intention dataset, achieving top marks in **Accuracy (0.8970)**, **AUC (0.9080)**, **Precision (0.7119)**, **F1 Score (0.6287)**, and **MCC (0.5751)**. E-commerce purchasing intent is governed by threshold triggers (e.g., visiting high-value pages right before checkout), which axis-aligned orthogonal tree splits capture substantially better than smooth hyperplanes or distance metrics. |

---

## Streamlit Web Application Features

The interactive web application (`app.py`) satisfies all four rubric capabilities:
1. **Dataset Upload Option (CSV)**: Users can upload unseen test CSV batches. If no custom file is uploaded, the app automatically loads the bundled 2,466-row test partition (`test_data.csv`).
2. **Model Selection Dropdown**: Seamlessly switch between any of the 5 trained classification models (`Decision Tree`, `Logistic Regression`, `Support Vector Machine`, `kNN`, `Naive Bayes`).
3. **Display of Evaluation Metrics**: Dynamic KPI cards displaying all 6 mandatory metrics: Accuracy, ROC-AUC, Precision, Recall, F1 Score, and MCC. Includes an interactive decision threshold slider.
4. **Confusion Matrix & Classification Report**: Rich visual confusion matrix with a Seaborn heatmap, side-by-side with an auto-styled precision/recall/support classification report.
5. **Bonus Capabilities**: Full prediction inspector with confidence scores, downloadable enriched CSV, and an integrated 5-model comparative benchmark table with color-coded winner highlights.

---

## Step-by-Step Deployment Instructions (Streamlit Community Cloud)

1. **Push to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Complete M.Tech ML Assignment 2: 5-Model Benchmark & Streamlit App"
   git branch -M main
   git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPO_NAME>.git
   git push -u origin main
   ```
2. **Deploy on Streamlit Community Cloud**:
   - Navigate to [https://streamlit.io/cloud](https://streamlit.io/cloud) and sign in using your GitHub account.
   - Click **"New App"** in the top right corner.
   - Select your repository (`<YOUR_REPO_NAME>`), set the branch to `main`, and specify the main file path as `app.py`.
   - Click **"Deploy"**. Streamlit Community Cloud will automatically parse `requirements.txt`, install dependencies, and launch your live interactive dashboard.
3. **Obtain Live Link**:
   - Copy your public app URL (e.g., `https://<YOUR_APP_SUBDOMAIN>.streamlit.app`) and verify that it opens smoothly in any browser.

---

## Verification & Local Reproduction

To run the pipeline and application locally:
```bash
# 1. Clone repository
git clone https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPO_NAME>.git
cd <YOUR_REPO_NAME>

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Re-train and evaluate all models from scratch
python model/train_models.py

# 4. Launch Streamlit web application
streamlit run app.py
```

