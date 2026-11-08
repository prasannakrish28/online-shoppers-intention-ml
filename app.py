"""
BITS Pilani - Work Integrated Learning Programmes (WILP)
M.Tech (Data Science and Engineering)
Course: Machine Learning (Assignment - 2)

Interactive Streamlit Application for Classification Model Benchmark & Deployment
Dataset: Online Shoppers Purchasing Intention (UCI Machine Learning Repository)
Target: Revenue (1 = Purchase, 0 = No Purchase)
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

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

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Online Shoppers Purchasing Intention - ML Classification",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 14px 18px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-val {
        font-size: 1.7rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Paths & File Constants
# ---------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(CURRENT_DIR, "model")
DEFAULT_TEST_CSV = os.path.join(CURRENT_DIR, "test_data.csv")
METRICS_SUMMARY_FILE = os.path.join(MODEL_DIR, "metrics_summary.json")

MODEL_FILES = {
    "Decision Tree": "decision_tree.joblib",
    "Logistic Regression": "logistic_regression.joblib",
    "Support Vector Machine": "svm.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib"
}

FEATURE_COLUMNS = [
    'Administrative', 'Administrative_Duration', 
    'Informational', 'Informational_Duration', 
    'ProductRelated', 'ProductRelated_Duration', 
    'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay',
    'Month', 'OperatingSystems', 'Browser', 
    'Region', 'TrafficType', 'VisitorType', 'Weekend'
]
TARGET_COLUMN = "Revenue"

# ---------------------------------------------------------
# Cached Loaders
# ---------------------------------------------------------
@st.cache_resource
def load_trained_pipeline(model_name: str):
    """Loads the serialized scikit-learn pipeline for the specified model."""
    filename = MODEL_FILES.get(model_name)
    if not filename:
        return None
    model_path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please run train_models.py first.")
        return None
    return joblib.load(model_path)

@st.cache_data
def load_default_data():
    """Loads bundled test data as default dataset."""
    if os.path.exists(DEFAULT_TEST_CSV):
        return pd.read_csv(DEFAULT_TEST_CSV)
    return None

@st.cache_data
def load_metrics_summary():
    """Loads pre-computed evaluation summary if available."""
    if os.path.exists(METRICS_SUMMARY_FILE):
        with open(METRICS_SUMMARY_FILE, "r") as f:
            return json.load(f)
    return None

# ---------------------------------------------------------
# Application Header
# ---------------------------------------------------------
st.markdown('<div class="main-title">Online Shoppers Purchasing Intention</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">BITS Pilani M.Tech (DSE) Machine Learning - Assignment 2 Benchmark Dashboard</div>',
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    
    # Dataset input
    st.subheader("Test Dataset")
    uploaded_file = st.file_uploader(
        "Upload Test Data (CSV)",
        type=["csv"],
        help="Upload held-out test data. If left blank, the bundled UCI test partition is utilized automatically."
    )
    
    # Model selection
    st.subheader("Model Selection")
    selected_model_name = st.selectbox(
        "Choose Classification Model:",
        options=list(MODEL_FILES.keys()),
        index=0,
        help="Select any of the 5 trained models to evaluate predictions and metrics."
    )
    
    # Decision threshold adjuster
    threshold = st.slider(
        "Decision Threshold (Probability)",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Adjust decision threshold for positive class probability (Revenue = 1)."
    )
    
    st.markdown("---")
    st.caption(
        "**Dataset**: UCI Online Shoppers Purchasing Intention\n\n"
        "**Target**: `Revenue` (1: Purchase, 0: No Purchase)\n\n"
        "**Features**: 17 (10 numerical, 7 categorical)\n\n"
        "**Course**: BITS Pilani WILP (M.Tech DSE)"
    )

# ---------------------------------------------------------
# Data Preparation
# ---------------------------------------------------------
if uploaded_file is not None:
    try:
        data_df = pd.read_csv(uploaded_file)
        data_source_label = f"Uploaded File ({uploaded_file.name})"
    except Exception as e:
        st.error(f"Error reading uploaded CSV: {e}")
        st.stop()
else:
    data_df = load_default_data()
    data_source_label = "Bundled UCI Test Split (`test_data.csv`)"

if data_df is None or data_df.empty:
    st.error("No test data found. Please upload a CSV file or check that `test_data.csv` exists in the repository.")
    st.stop()

# Validate features
has_target = TARGET_COLUMN in data_df.columns
missing_features = [col for col in FEATURE_COLUMNS if col not in data_df.columns]
if missing_features:
    st.warning(f"Note: Uploaded data is missing expected feature columns: {missing_features}. Inference may be impacted.")

X_eval = data_df[[c for c in FEATURE_COLUMNS if c in data_df.columns]]
y_true = data_df[TARGET_COLUMN].astype(int) if has_target else None

# ---------------------------------------------------------
# Inference & Metric Computation
# ---------------------------------------------------------
pipeline = load_trained_pipeline(selected_model_name)

if pipeline is None:
    st.error(f"Could not load pipeline for '{selected_model_name}'.")
    st.stop()

# Generate probability and class predictions
try:
    if hasattr(pipeline, "predict_proba"):
        y_prob = pipeline.predict_proba(X_eval)[:, 1]
    elif hasattr(pipeline, "decision_function"):
        raw_scores = pipeline.decision_function(X_eval)
        # Apply sigmoid normalization for thresholding
        y_prob = 1 / (1 + np.exp(-raw_scores))
    else:
        y_prob = pipeline.predict(X_eval)
    
    # Custom thresholding
    y_pred = (y_prob >= threshold).astype(int)
except Exception as e:
    st.error(f"Prediction failed on input data: {e}")
    st.stop()

# Compute metrics if ground truth target is present
metrics = {}
if has_target and len(np.unique(y_true)) > 1:
    metrics["Accuracy"] = accuracy_score(y_true, y_pred)
    try:
        metrics["AUC"] = roc_auc_score(y_true, y_prob)
    except Exception:
        metrics["AUC"] = 0.5
    metrics["Precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["Recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["F1"] = f1_score(y_true, y_pred, zero_division=0)
    metrics["MCC"] = matthews_corrcoef(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
else:
    cm = None
    report = None

# ---------------------------------------------------------
# Evaluation Metrics Display
# ---------------------------------------------------------
st.markdown(f"### Evaluation Metrics: `{selected_model_name}`")
st.caption(f"Evaluated on: **{data_source_label}** - {len(data_df):,} instances | Decision Threshold = `{threshold:.2f}`")

if metrics:
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">Accuracy</div>
            <div class="metric-val">{metrics['Accuracy']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">AUC Score</div>
            <div class="metric-val">{metrics['AUC']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">Precision</div>
            <div class="metric-val">{metrics['Precision']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">Recall</div>
            <div class="metric-val">{metrics['Recall']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    with c5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">F1 Score</div>
            <div class="metric-val">{metrics['F1']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    with c6:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">MCC Score</div>
            <div class="metric-val">{metrics['MCC']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("Uploaded dataset does not contain ground truth `Revenue` column. Generating predictions only.")

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Tabbed Analysis: Confusion Matrix, Report, Data & Benchmark
# ---------------------------------------------------------
tab_eval, tab_inspect, tab_benchmark, tab_about = st.tabs([
    "Confusion Matrix & Report",
    "Sample Predictions",
    "Model Benchmark Comparison",
    "Dataset & Methodology"
])

# ---------------------------------------------------------
# Tab 1: Confusion Matrix & Classification Report
# ---------------------------------------------------------
with tab_eval:
    if cm is not None and report is not None:
        col_cm, col_rep = st.columns([1, 1])
        
        with col_cm:
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                xticklabels=["No Revenue (0)", "Revenue (1)"],
                yticklabels=["No Revenue (0)", "Revenue (1)"],
                ax=ax,
                annot_kws={"size": 13, "weight": "bold"}
            )
            ax.set_xlabel("Predicted Label", fontweight="bold", fontsize=10)
            ax.set_ylabel("True Label", fontweight="bold", fontsize=10)
            ax.set_title(f"Confusion Matrix: {selected_model_name}", fontsize=11, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            
            st.caption(
                f"**True Negatives (TN)**: {cm[0][0]:,} | **False Positives (FP)**: {cm[0][1]:,}\n\n"
                f"**False Negatives (FN)**: {cm[1][0]:,} | **True Positives (TP)**: {cm[1][1]:,}"
            )
            
        with col_rep:
            st.subheader("Classification Report")
            rep_df = pd.DataFrame(report).transpose()
            
            # Formatting table display
            st.dataframe(
                rep_df.style.format({
                    "precision": "{:.4f}",
                    "recall": "{:.4f}",
                    "f1-score": "{:.4f}",
                    "support": "{:.0f}"
                }).background_gradient(cmap="Blues", subset=["precision", "recall", "f1-score"]),
                use_container_width=True
            )
            
            st.markdown("""
            **Metric Guidance for Imbalanced Class Analysis**:
            - **Accuracy**: Overall fraction of correct predictions across both classes.
            - **Precision**: When model predicts a purchase, how often is it right?
            - **Recall (Sensitivity)**: What proportion of actual purchasers were successfully captured?
            - **F1 Score**: Harmonic mean of Precision and Recall.
            - **MCC (Matthews Correlation Coefficient)**: Balanced metric in $[-1, +1]$ taking into account all 4 confusion matrix cells.
            """)
    else:
        st.info("Confusion matrix and classification report require ground truth `Revenue` column.")

# ---------------------------------------------------------
# Tab 2: Sample Predictions Inspection
# ---------------------------------------------------------
with tab_inspect:
    st.subheader("Test Data Predictions Inspector")
    st.caption("Review individual session records alongside predicted purchase probability and binary classification.")
    
    inspect_df = X_eval.copy()
    if has_target:
        inspect_df["Actual_Revenue"] = y_true
    inspect_df["Predicted_Revenue"] = y_pred
    inspect_df["Purchase_Probability"] = np.round(y_prob, 4)
    
    if has_target:
        inspect_df["Prediction_Match"] = inspect_df["Actual_Revenue"] == inspect_df["Predicted_Revenue"]
    
    st.dataframe(inspect_df.head(100), use_container_width=True)
    
    # Download predictions
    csv_bytes = inspect_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Predictions (CSV)",
        data=csv_bytes,
        file_name=f"predictions_{selected_model_name.lower().replace(' ', '_')}.csv",
        mime="text/csv"
    )

# ---------------------------------------------------------
# Tab 3: Model Benchmark Comparison Table
# ---------------------------------------------------------
with tab_benchmark:
    st.subheader("Comparative Performance Across All 5 ML Models")
    st.markdown("All 5 classification models evaluated under identical train-test conditions (80% train, 20% stratified test split, random_state=42).")
    
    summary = load_metrics_summary()
    if summary:
        rows = []
        for name, data in summary.items():
            m = data["metrics"]
            rows.append({
                "ML Model Name": name,
                "Accuracy": m["Accuracy"],
                "AUC": m["AUC"],
                "Precision": m["Precision"],
                "Recall": m["Recall"],
                "F1": m["F1"],
                "MCC": m["MCC"]
            })
        benchmark_table = pd.DataFrame(rows).set_index("ML Model Name")
        
        # Highlight best model
        st.dataframe(
            benchmark_table.style.format("{:.4f}")
            .highlight_max(axis=0, color="#D1FAE5")
            .highlight_min(axis=0, color="#FEE2E2"),
            use_container_width=True
        )
        
        st.success(
            "**Best Performing Model: Decision Tree Classifier**\n\n"
            "- **Accuracy**: 0.8970\n"
            "- **AUC**: 0.9080\n"
            "- **Precision**: 0.7119\n"
            "- **F1 Score**: 0.6287\n"
            "- **MCC**: 0.5751\n\n"
            "The Decision Tree effectively captures hierarchical decision boundaries on features like PageValues "
            "and ProductRelated_Duration without suffering from linear boundary limitations or class density distortion."
        )
    else:
        st.info("Metrics summary file not found. Run `train_models.py` to compile the comparative table.")

# ---------------------------------------------------------
# Tab 4: Dataset & Methodology Description
# ---------------------------------------------------------
with tab_about:
    st.subheader("About the Dataset & Preprocessing Pipeline")
    st.markdown("""
    ### 1. Dataset Overview
    - **Repository**: [UCI Machine Learning Repository - Online Shoppers Purchasing Intention](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
    - **Total Instances**: 12,330 e-commerce browsing sessions
    - **Total Features**: 17 features (10 numerical, 7 categorical / boolean)
    - **Target Variable**: `Revenue` (True / False) - indicates whether the user completed a purchase transaction.
    - **Class Balance**: 84.5% non-purchasing sessions (`Revenue = False`), 15.5% purchasing sessions (`Revenue = True`).
    
    ### 2. Feature Definitions
    - **Administrative, Informational, ProductRelated**: Number of pages visited in each category during the session.
    - **Administrative_Duration, Informational_Duration, ProductRelated_Duration**: Total duration (seconds) spent on pages of each type.
    - **BounceRates**: Percentage of visitors who enter the site and leave without triggering any other requests.
    - **ExitRates**: Proportion of pageviews to the site that were the last in the session.
    - **PageValues**: Average value of a web page that a user visited before completing an e-commerce transaction.
    - **SpecialDay**: Closeness of the browsing date to a specific special day (e.g., Mother's Day, Valentine's Day).
    - **Month, OperatingSystems, Browser, Region, TrafficType, VisitorType, Weekend**: Session metadata and client attributes.
    
    ### 3. Preprocessing Architecture
    To eliminate data leakage while ensuring reproducible inference:
    - Numerical features are standardized using `StandardScaler`.
    - Categorical variables (`Month`, `VisitorType`, etc.) are encoded using `OneHotEncoder(handle_unknown='ignore')`.
    - The preprocessing transform and classifier are coupled into a unified `Pipeline`, persisted with `joblib`.
    """)

