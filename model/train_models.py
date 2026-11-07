"""
BITS Pilani - Work Integrated Learning Programmes (WILP)
M.Tech (Data Science and Engineering) - Machine Learning
Assignment - 2: Classification Model Benchmark & Deployment

Dataset: Online Shoppers Purchasing Intention (UCI Machine Learning Repository)
Source: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
Target: Revenue (1 = Purchase, 0 = No Purchase)
"""

import os
import io
import json
import zipfile
import urllib.request
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix
)

# Configuration & Paths
RANDOM_STATE = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_URL = "https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip"

def download_and_load_dataset():
    """
    Downloads and extracts the Online Shoppers Purchasing Intention dataset from UCI.
    Returns: pandas DataFrame
    """
    print(f"Downloading dataset from UCI Machine Learning Repository...")
    req = urllib.request.Request(
        DATA_URL,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
        csv_file = [name for name in archive.namelist() if name.endswith('.csv')][0]
        df = pd.read_csv(archive.open(csv_file))
    print(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns.")
    return df

def build_preprocessor(numerical_features, categorical_features):
    """
    Creates a scikit-learn ColumnTransformer that standardizes numerical features
    and one-hot encodes categorical features.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )
    return preprocessor

def evaluate_model(model_name, pipeline, X_test, y_test):
    """
    Computes all 6 assignment-mandated classification metrics:
    1. Accuracy
    2. AUC Score (ROC-AUC)
    3. Precision
    4. Recall
    5. F1 Score
    6. Matthews Correlation Coefficient (MCC Score)
    """
    y_pred = pipeline.predict(X_test)
    
    # Calculate probability scores for ROC-AUC
    if hasattr(pipeline, "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)[:, 1]
    elif hasattr(pipeline, "decision_function"):
        y_prob = pipeline.decision_function(X_test)
    else:
        y_prob = y_pred

    metrics = {
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "AUC": float(roc_auc_score(y_test, y_prob)),
        "Precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "F1": float(f1_score(y_test, y_pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_test, y_pred))
    }
    
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    print(f"\n--- {model_name} ---")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f}")
        
    return metrics, cm, report

def main():
    # 1. Load Data
    df = download_and_load_dataset()
    
    # Target definition: Revenue (boolean -> binary 0/1)
    target_col = "Revenue"
    df[target_col] = df[target_col].astype(int)
    
    # Identify feature columns
    numerical_cols = [
        'Administrative', 'Administrative_Duration', 
        'Informational', 'Informational_Duration', 
        'ProductRelated', 'ProductRelated_Duration', 
        'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay'
    ]
    categorical_cols = [
        'Month', 'OperatingSystems', 'Browser', 
        'Region', 'TrafficType', 'VisitorType', 'Weekend'
    ]
    
    feature_cols = numerical_cols + categorical_cols
    X = df[feature_cols]
    y = df[target_col]
    
    # 2. Stratified Train-Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set:     {X_test.shape[0]} samples")
    print(f"Positive class ratio (Test): {y_test.mean():.4f}")
    
    # Save test dataset as test_data.csv in project root
    test_df = X_test.copy()
    test_df[target_col] = y_test
    test_csv_path = os.path.join(PROJECT_ROOT, "test_data.csv")
    test_df.to_csv(test_csv_path, index=False)
    print(f"Saved test dataset to: {test_csv_path}")
    
    # Also save a small sample of test data (e.g. 500 rows) as test_sample.csv for fast testing
    sample_csv_path = os.path.join(PROJECT_ROOT, "test_sample.csv")
    test_df.head(500).to_csv(sample_csv_path, index=False)
    
    # 3. Model Definitions
    # Notice: Each model is coupled with preprocessor in a clean Scikit-Learn Pipeline
    # to avoid data leakage and enable seamless inference on unseen CSV inputs.
    preprocessor = build_preprocessor(numerical_cols, categorical_cols)
    
    models = {
        "Logistic Regression": Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced'))
        ]),
        "Decision Tree": Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', DecisionTreeClassifier(max_depth=6, min_samples_split=10, random_state=RANDOM_STATE))
        ]),
        "kNN": Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', KNeighborsClassifier(n_neighbors=7, weights='distance'))
        ]),
        "Naive Bayes": Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', GaussianNB())
        ]),
        "Support Vector Machine": Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', SVC(kernel='rbf', C=1.0, probability=True, random_state=RANDOM_STATE))
        ])
    }
    
    # Model filenames mapping
    model_filenames = {
        "Logistic Regression": "logistic_regression.joblib",
        "Decision Tree": "decision_tree.joblib",
        "kNN": "knn.joblib",
        "Naive Bayes": "naive_bayes.joblib",
        "Support Vector Machine": "svm.joblib"
    }
    
    all_metrics = {}
    evaluation_details = {}
    
    # 4. Train and Evaluate each model
    for name, pipeline in models.items():
        print(f"\nTraining {name}...")
        pipeline.fit(X_train, y_train)
        
        # Save trained pipeline model into model/ directory
        model_save_path = os.path.join(BASE_DIR, model_filenames[name])
        joblib.dump(pipeline, model_save_path)
        print(f"Saved model artifact: {model_save_path}")
        
        # Evaluate on test set
        m, cm, report = evaluate_model(name, pipeline, X_test, y_test)
        all_metrics[name] = m
        evaluation_details[name] = {
            "metrics": m,
            "confusion_matrix": cm,
            "classification_report": report
        }
        
    # 5. Save Summary Metrics JSON for Web App
    summary_path = os.path.join(BASE_DIR, "metrics_summary.json")
    with open(summary_path, "w") as f:
        json.dump(evaluation_details, f, indent=4)
    print(f"\nSaved metrics summary to: {summary_path}")
    
    # 6. Display Comparison Table
    comparison_df = pd.DataFrame(all_metrics).T
    comparison_df = comparison_df[["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]]
    print("\n================ FINAL COMPARISON TABLE ================")
    print(comparison_df.round(4).to_string())
    print("========================================================\n")

if __name__ == "__main__":
    main()
