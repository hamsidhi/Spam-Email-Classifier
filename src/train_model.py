"""
Grand Line Message Bounty Detector — One Piece Anime Themed Spam Classifier
Script: train_model.py
Description: Trains a Marine Spam Classifier using a Devil Fruit TF-IDF Vectorizer
             and Naive Bayes pipeline on the Grand Line SMS message dataset.
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


def locate_grand_line_paths():
    """
    Locates project root and target file paths relative to script location.
    Works seamlessly across Windows PowerShell sessions.
    """
    script_location = Path(__file__).resolve().parent
    grand_line_root = script_location.parent
    bounty_dataset_path = grand_line_root / "data" / "SMSSpamCollection"
    marine_hq_models_dir = grand_line_root / "models"
    saved_pipeline_path = marine_hq_models_dir / "spam_message_pipeline.joblib"
    
    return grand_line_root, bounty_dataset_path, marine_hq_models_dir, saved_pipeline_path


def inspect_bounty_scrolls(bounty_dataset_path):
    """
    Loads tab-separated SMS dataset, checks for missing data,
    and displays message distributions.
    """
    print("\n" + "=" * 65)
    print("      GRAND LINE MARINE HEADQUARTERS — BOUNTY SCROLL LOADING      ")
    print("=" * 65)
    
    # 1. Verify dataset file existence
    if not bounty_dataset_path.exists():
        print(f"\n[ERROR] Bounty scroll missing at target path: {bounty_dataset_path}")
        print("[INSTRUCTION] Please place 'SMSSpamCollection' inside data/ directory.")
        print("              Refer to data/README.md for download steps.\n")
        sys.exit(1)
        
    print(f"[INFO] Found bounty message scroll at: {bounty_dataset_path}")
    
    # 2. Read dataset with fallback encoding
    try:
        scrolls_df = pd.read_csv(
            bounty_dataset_path,
            sep="\t",
            header=None,
            names=["label", "message"],
            encoding="utf-8"
        )
    except UnicodeDecodeError:
        print("[WARN] UTF-8 encoding issue detected. Retrying with 'latin-1' encoding...")
        scrolls_df = pd.read_csv(
            bounty_dataset_path,
            sep="\t",
            header=None,
            names=["label", "message"],
            encoding="latin-1"
        )

    # 3. Display scroll statistics
    total_scrolls = len(scrolls_df)
    print(f"[INFO] Total bounty message scrolls loaded: {total_scrolls}")
    
    print("\n--- Preview of First 5 Bounty Message Scrolls ---")
    print(scrolls_df.head())
    
    missing_scrolls = scrolls_df.isnull().sum()
    print("\n--- Missing Value Check ---")
    print(missing_scrolls)
    
    if missing_scrolls.sum() > 0:
        print("[INFO] Purging corrupted message scrolls...")
        scrolls_df = scrolls_df.dropna(subset=["label", "message"])
        print(f"[INFO] Remaining valid message scrolls: {len(scrolls_df)}")
    else:
        print("[INFO] All message scrolls verified cleanly! Zero missing values.")
        
    # 4. Display label distribution
    bounty_counts = scrolls_df["label"].value_counts()
    print("\n--- Message Label Proportions ---")
    for bounty_type, count in bounty_counts.items():
        pct = (count / len(scrolls_df)) * 100
        tag = "Crewmate Message (ham)" if bounty_type == "ham" else "Pirate Spam Notice (spam)"
        print(f"  - {bounty_type:<6} ({tag}): {count:5d} ({pct:.2f}%)")
        
    return scrolls_df


def train_marine_fleet_pipeline(crew_train_messages, crew_train_labels):
    """
    Builds and trains a scikit-learn Pipeline containing:
      1. Devil Fruit Vectorizer (TfidfVectorizer)
      2. Marine Classifier (MultinomialNB)
    Fits strictly on training data to prevent data leakage.
    """
    print("\n" + "=" * 65)
    print("    GRAND LINE MARINE CLASSIFIER — DEVIL FRUIT PIPELINE FIT    ")
    print("=" * 65)
    print("[INFO] Initializing Devil Fruit Vectorizer & Marine Naive Bayes Classifier...")
    
    # Unified scikit-learn Pipeline
    devil_fruit_vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        sublinear_tf=True
    )
    marine_classifier = MultinomialNB()
    
    marine_spam_pipeline = Pipeline([
        ("devil_fruit_vectorizer", devil_fruit_vectorizer),
        ("marine_classifier", marine_classifier)
    ])
    
    print("[INFO] Training Marine Fleet Pipeline on training split...")
    marine_spam_pipeline.fit(crew_train_messages, crew_train_labels)
    print("[SUCCESS] Marine Fleet Pipeline successfully trained!")
    
    return marine_spam_pipeline


def evaluate_battle_results(marine_spam_pipeline, crew_test_messages, crew_test_labels):
    """
    Evaluates trained pipeline performance on unseen test split.
    """
    print("\n" + "=" * 65)
    print("      GRAND LINE BATTLE RESULTS — PERFORMANCE EVALUATION      ")
    print("=" * 65)
    
    # Generate predictions on unseen test split
    predicted_bounties = marine_spam_pipeline.predict(crew_test_messages)
    
    # Compute metrics (positive class = 'spam')
    accuracy = accuracy_score(crew_test_labels, predicted_bounties)
    precision = precision_score(crew_test_labels, predicted_bounties, pos_label="spam")
    recall = recall_score(crew_test_labels, predicted_bounties, pos_label="spam")
    f1 = f1_score(crew_test_labels, predicted_bounties, pos_label="spam")
    
    print("\n--- Marine Battle Performance Summary ---")
    print(f"  Accuracy Score : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"  Spam Precision : {precision:.4f} ({precision * 100:.2f}%) [Target: 'spam']")
    print(f"  Spam Recall    : {recall:.4f} ({recall * 100:.2f}%) [Target: 'spam']")
    print(f"  Spam F1-Score  : {f1:.4f} ({f1 * 100:.2f}%) [Target: 'spam']")
    
    print("\n--- Classification Detail Report ---")
    print(classification_report(crew_test_labels, predicted_bounties, target_names=["ham", "spam"]))
    
    print("--- Marine Battle Confusion Matrix ---")
    matrix = confusion_matrix(crew_test_labels, predicted_bounties, labels=["ham", "spam"])
    print(f"                 Predicted 'ham'   Predicted 'spam'")
    print(f"Actual 'ham'  :       {matrix[0][0]:<15} {matrix[0][1]:<15}")
    print(f"Actual 'spam' :       {matrix[1][0]:<15} {matrix[1][1]:<15}")
    print("=" * 65)


def save_marine_den_den_mushi(marine_spam_pipeline, saved_pipeline_path):
    """
    Serializes fitted pipeline to disk using joblib.
    """
    saved_pipeline_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(marine_spam_pipeline, saved_pipeline_path)
    print(f"\n[SUCCESS] Marine Spam Pipeline artifact saved to: {saved_pipeline_path}\n")


def main():
    # 1. Resolve paths
    grand_line_root, bounty_dataset_path, marine_hq_models_dir, saved_pipeline_path = locate_grand_line_paths()
    
    # 2. Inspect and load data
    scrolls_df = inspect_bounty_scrolls(bounty_dataset_path)
    
    # 3. Separate message content and label targets
    grand_line_messages = scrolls_df["message"]
    bounty_labels = scrolls_df["label"]
    
    # 4. Stratified 80/20 train/test split
    print("\n[INFO] Dividing Grand Line messages into 80% train crew and 20% test crew...")
    (
        crew_train_messages,
        crew_test_messages,
        crew_train_labels,
        crew_test_labels
    ) = train_test_split(
        grand_line_messages,
        bounty_labels,
        test_size=0.20,
        random_state=42,
        stratify=bounty_labels
    )
    
    print(f"[INFO] Training crew count : {len(crew_train_messages)}")
    print(f"[INFO] Testing crew count  : {len(crew_test_messages)}")
    
    # 5. Train pipeline
    marine_spam_pipeline = train_marine_fleet_pipeline(crew_train_messages, crew_train_labels)
    
    # 6. Evaluate on unseen test crew
    evaluate_battle_results(marine_spam_pipeline, crew_test_messages, crew_test_labels)
    
    # 7. Save trained artifact
    save_marine_den_den_mushi(marine_spam_pipeline, saved_pipeline_path)


if __name__ == "__main__":
    main()
