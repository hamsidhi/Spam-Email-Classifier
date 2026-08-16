"""
Grand Line Message Bounty Detector — Spam Message Classifier
Script: train_model.py
Description: Trains a baseline SMS spam classifier using a TF-IDF vectorizer
             and Multinomial Naive Bayes pipeline on the UCI SMS Spam Collection dataset.
"""

import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import joblib


def get_project_paths():
    """
    Resolves project directories dynamically relative to this script's location.
    Ensures seamless execution on Windows across different working directories.
    """
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_path = project_root / "data" / "SMSSpamCollection"
    models_dir = project_root / "models"
    model_save_path = models_dir / "spam_message_pipeline.joblib"
    
    return project_root, data_path, models_dir, model_save_path


def load_and_inspect_dataset(data_path):
    """
    Loads tab-separated SMS spam dataset, validates columns, checks for missing data,
    and displays fundamental dataset characteristics.
    """
    print("\n" + "=" * 60)
    print("      GRAND LINE MARINE INTELLIGENCE — DATASET LOADING      ")
    print("=" * 60)
    
    # 1. Verify dataset existence
    if not data_path.exists():
        print(f"\n[ERROR] Dataset file missing at target path: {data_path}")
        print("[INSTRUCTION] Please manually place 'SMSSpamCollection' in the data/ folder.")
        print("              Refer to data/README.md for download instructions.\n")
        sys.exit(1)
        
    print(f"[INFO] Found dataset file at: {data_path}")
    
    # 2. Load tab-separated file with fallback encoding
    try:
        df = pd.read_csv(
            data_path,
            sep="\t",
            header=None,
            names=["label", "message"],
            encoding="utf-8"
        )
    except UnicodeDecodeError:
        print("[WARN] UTF-8 decoding issue encountered. Retrying with 'latin-1' encoding...")
        df = pd.read_csv(
            data_path,
            sep="\t",
            header=None,
            names=["label", "message"],
            encoding="latin-1"
        )

    # 3. Inspect dataset properties
    total_rows = len(df)
    print(f"[INFO] Total raw messages loaded: {total_rows}")
    
    print("\n--- First 5 Dataset Records ---")
    print(df.head())
    
    missing_count = df.isnull().sum()
    print("\n--- Missing Value Check ---")
    print(missing_count)
    
    # 4. Clean missing or invalid rows if any exist
    if missing_count.sum() > 0:
        print("[INFO] Cleaning missing or empty text records...")
        df = df.dropna(subset=["label", "message"])
        print(f"[INFO] Rows remaining after cleaning: {len(df)}")
    else:
        print("[INFO] Data integrity verified: No missing values found.")
        
    # 5. Label validation and distribution
    class_counts = df["label"].value_counts()
    print("\n--- Class Label Distribution ---")
    for label, count in class_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  - {label:<8}: {count:5d} ({percentage:.2f}%)")
        
    valid_labels = {"ham", "spam"}
    dataset_labels = set(df["label"].unique())
    if not dataset_labels.issubset(valid_labels):
        print(f"\n[ERROR] Unexpected class labels found: {dataset_labels}. Expected only 'ham' and 'spam'.")
        sys.exit(1)
        
    return df


def build_and_train_pipeline(X_train, y_train):
    """
    Constructs and fits a scikit-learn Pipeline combining TfidfVectorizer and MultinomialNB.
    Fitting occurs strictly on training data to prevent data leakage.
    """
    print("\n" + "=" * 60)
    print("     GRAND LINE MARINE CLASSIFIER — PIPELINE TRAINING     ")
    print("=" * 60)
    print("[INFO] Initializing TF-IDF Vectorizer & Multinomial Naive Bayes Pipeline...")
    
    # Construct scikit-learn Pipeline
    pipeline = Pipeline([
        (
            "vectorizer",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                sublinear_tf=True
            )
        ),
        (
            "classifier",
            MultinomialNB()
        )
    ])
    
    print("[INFO] Training Marine Classifier on training split...")
    pipeline.fit(X_train, y_train)
    print("[SUCCESS] Pipeline training complete!")
    
    return pipeline


def evaluate_marine_classifier(pipeline, X_test, y_test):
    """
    Evaluates trained pipeline performance on unseen test data across standard metrics.
    """
    print("\n" + "=" * 60)
    print("      GRAND LINE MARINE BATTLE RESULTS — MODEL EVALUATION    ")
    print("=" * 60)
    
    # Generate test predictions
    y_pred = pipeline.predict(X_test)
    
    # Compute quantitative metrics (with 'spam' as positive target label)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label="spam")
    rec = recall_score(y_test, y_pred, pos_label="spam")
    f1 = f1_score(y_test, y_pred, pos_label="spam")
    
    print("\n--- Baseline Performance Summary ---")
    print(f"  Accuracy Score : {acc:.4f} ({acc * 100:.2f}%)")
    print(f"  Precision Score: {prec:.4f} ({prec * 100:.2f}%) [Positive Class: 'spam']")
    print(f"  Recall Score   : {rec:.4f} ({rec * 100:.2f}%) [Positive Class: 'spam']")
    print(f"  F1-Score       : {f1:.4f} ({f1 * 100:.2f}%) [Positive Class: 'spam']")
    
    print("\n--- Detailed Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))
    
    print("--- Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])
    print(f"                 Predicted 'ham'   Predicted 'spam'")
    print(f"Actual 'ham'  :       {cm[0][0]:<15} {cm[0][1]:<15}")
    print(f"Actual 'spam' :       {cm[1][0]:<15} {cm[1][1]:<15}")
    print("=" * 60)


def save_trained_pipeline(pipeline, model_save_path):
    """
    Saves trained scikit-learn pipeline to disk using joblib.
    """
    # Ensure models directory exists
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(pipeline, model_save_path)
    print(f"\n[SUCCESS] Saved Marine Spam Pipeline to: {model_save_path}\n")


def main():
    # 1. Resolve directory paths
    project_root, data_path, models_dir, model_save_path = get_project_paths()
    
    # 2. Load & inspect dataset
    df = load_and_inspect_dataset(data_path)
    
    # 3. Separate features and target
    X = df["message"]
    y = df["label"]
    
    # 4. Stratified 80/20 train/test split (fixed random_state=42)
    print("\n[INFO] Performing 80/20 train/test split (stratified by target label)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    print(f"[INFO] Training samples : {len(X_train)}")
    print(f"[INFO] Testing samples  : {len(X_test)}")
    
    # 5. Build and train pipeline
    pipeline = build_and_train_pipeline(X_train, y_train)
    
    # 6. Evaluate pipeline on test set
    evaluate_marine_classifier(pipeline, X_test, y_test)
    
    # 7. Save pipeline artifact
    save_trained_pipeline(pipeline, model_save_path)


if __name__ == "__main__":
    main()
