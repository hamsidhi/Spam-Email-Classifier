# Grand Line Message Bounty Detector — SMS Spam Message Classifier

An approachable, beginner-friendly Natural Language Processing (NLP) machine learning project for detecting spam text messages using scikit-learn, TF-IDF vectorization, and Multinomial Naive Bayes. Designed with a light, presentation-friendly *One Piece* theme for internship submission.

---

## Project Overview

The **Grand Line Message Bounty Detector** is a binary text classification application. It processes short text messages and predicts whether a given message is **`ham`** (a legitimate message) or **`spam`** (an unwanted, promotional, or fraudulent message).

The core of the system is built around a unified scikit-learn `Pipeline` that combines Term Frequency-Inverse Document Frequency (`TfidfVectorizer`) text feature extraction with a probabilistic `MultinomialNB` classifier.

---

## Important Scope Clarification

- **SMS vs. Email Dataset:** The dataset used in this project is the **UCI SMS Spam Collection**, which consists of short SMS text messages. This project is technically an **SMS Spam Message Classifier**.
- **Scope Distinction:** Full email security systems analyze MIME headers (e.g., `From:`, `Subject:`, `DKIM-Signature`), HTML layout structure, embedded hyperlink domains, network routing metadata, and file attachments. Plain SMS text messages lack these complex structural components.
- **Internship Alignment:** The internship project is titled *"Spam Email Classifier"*. This implementation focuses on the core natural language text-classification engine, which serves as the foundational text-processing module for email body analysis.

---

## Features

- **Automated Dataset Validation:** Inspects raw tab-separated records, verifies column structures, checks for missing values, and calculates class distributions.
- **Data Leakage Prevention:** Uses a stratified 80/20 train/test split (`random_state=42`) and fits the text vectorizer exclusively on training data inside a scikit-learn `Pipeline`.
- **TF-IDF Feature Extraction:** Applies sublinear term frequency scaling and Unicode accent normalization to highlight informative spam keywords.
- **Multinomial Naive Bayes Classification:** Leverages a lightweight, highly efficient probabilistic model optimized for sparse word counts.
- **Comprehensive Model Evaluation:** Computes Accuracy, Precision, Recall, F1-score (with `spam` as the positive class), a detailed classification report, and a formatted 2x2 confusion matrix.
- **Model Persistence:** Serializes the complete trained pipeline into a single `.joblib` file (`models/spam_message_pipeline.joblib`).
- **Interactive Inspection CLI:** Provides a user-friendly terminal interface for testing custom input strings in real time.

---

## Architecture Diagram

```markdown
SMSSpamCollection
        ↓
Data Loading & Validation (pandas, tab-separated format)
        ↓
Stratified Train/Test Split (80% train, 20% test, random_state=42)
        ↓
TF-IDF Vectorization (fitted on train split only via Pipeline)
        ↓
Multinomial Naive Bayes (classifier fit)
        ↓
Evaluation on Unseen Test Data (accuracy, precision, recall, f1, confusion matrix)
        ↓
Saved Model Pipeline (models/spam_message_pipeline.joblib)
        ↓
Interactive Message Prediction (src/predict_message.py CLI)
```

---

## Project Structure

```
Project-1-Spam-Email-Classifier/
│
├── data/
│   ├── SMSSpamCollection           # Raw dataset file (manually placed, git-ignored)
│   └── README.md                   # Dataset details and download instructions
│
├── models/
│   ├── .gitkeep                    # Tracks models directory in Git
│   └── spam_message_pipeline.joblib # Serialized trained pipeline (generated via train)
│
├── reports/
│   └── project_report.md           # Comprehensive technical internship report
│
├── src/
│   ├── train_model.py              # Loads data, trains pipeline, evaluates, saves artifact
│   └── predict_message.py          # Interactive CLI for testing custom messages
│
├── .gitignore                      # Ignores byte-code, virtual envs, datasets, & joblib files
├── README.md                       # Main project documentation (this file)
└── requirements.txt                # Lightweight Python package dependencies
```

### Explanation of Repository Files:
- **`data/SMSSpamCollection`**: The raw dataset file containing 5,574 physical lines. After parsing expected tab-separated `label` and `message` fields, the training script loaded 5,572 valid records (4,825 `ham` and 747 `spam`).
- **`data/README.md`**: Download and setup instructions for obtaining the raw dataset.
- **`src/train_model.py`**: Execution script for data inspection, splitting, training, evaluation, and pipeline saving.
- **`src/predict_message.py`**: Interactive console application for checking single text inputs against the saved model.
- **`models/`**: Folder holding model binaries (`.joblib`).
- **`reports/`**: Destination folder for detailed project reports.
- **`requirements.txt`**: Minimal list of Python dependencies (`pandas`, `scikit-learn`, `joblib`).
- **`.gitignore`**: Excludes temporary files, local environments, datasets, and generated models.

---

## Dataset & Schema

- **Source:** [UCI Machine Learning Repository - SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
- **Creators:** Tiago A. Almeida and José María Gómez Hidalgo.
- **Size & Parsing:** The downloaded source file contains 5,574 physical lines. After parsing the expected tab-separated `label` and `message` fields, the training script loaded 5,572 valid records: 4,825 `ham` messages and 747 `spam` messages. The experiment and all reported metrics are based on these 5,572 parsed records.
- **Format:** Tab-separated text (`label\tmessage`), encoded in UTF-8 / Latin-1.
- **Setup Note:** The raw dataset file (`SMSSpamCollection`) must be placed manually in `data/SMSSpamCollection` and is ignored by Git according to `.gitignore`.

### Data Schema Table

| Column Name | Data Type | Description / Allowed Values |
|---|---|---|
| `label` | String | Target label: `ham` (legitimate) or `spam` (unwanted/fraudulent) |
| `message` | String | Raw text content of the SMS message |

---

## Technologies Used

- **Python (3.8+)**: Core programming language.
- **pandas**: Tabular data manipulation and CSV parsing.
- **scikit-learn**: Machine learning tools (`TfidfVectorizer`, `MultinomialNB`, `Pipeline`, metrics, `train_test_split`).
- **joblib**: Efficient serialization of scikit-learn pipelines.
- **pathlib**: Cross-platform object-oriented filesystem path resolution.

---

## Machine Learning Workflow

1. **Loading:** Reads tab-separated file with fallback encoding support (`utf-8` then `latin-1`).
2. **Validation:** Checks row counts, verifies 0 missing values, and checks class balance (~86.6% `ham`, ~13.4% `spam`).
3. **Splitting:** Divides data into 80% training (`X_train`, `y_train`) and 20% test (`X_test`, `y_test`) sets using `stratify=y` and `random_state=42`.
4. **Vectorization:** Transforms raw strings into TF-IDF numerical feature matrices. The vectorizer is **fit only on training data** inside the pipeline to prevent data leakage.
5. **Training:** Fits the `MultinomialNB` classifier on feature vectors derived from training data.
6. **Evaluation:** Computes metrics on unseen test data (`X_test`).
7. **Persistence:** Saves the unified fitted pipeline artifact containing both vectorizer vocabulary and classifier parameters to `models/spam_message_pipeline.joblib`.
8. **Prediction:** Loads the saved pipeline and transforms new input strings using `transform()` (never `fit_transform()`).

---

## Evaluation Metrics Explained

- **Accuracy:** Overall proportion of correctly predicted messages out of total test messages. Note: Accuracy alone can be misleading due to class imbalance (~86.6% `ham` vs. ~13.4% `spam`).
- **Precision (Positive class = `spam`):** Out of all messages predicted as `spam`, how many were actually spam? (Measures false alarm rate).
- **Recall (Positive class = `spam`):** Out of all actual `spam` messages in the test set, how many did the model correctly detect? (Measures catch rate).
- **F1-Score:** The harmonic mean of precision and recall, serving as a balanced quality measure.
- **Confusion Matrix:** 2x2 table mapping actual vs. predicted classes:
  - **True Negative (TN):** Actual `ham` correctly classified as `ham`.
  - **False Positive (FP):** Actual `ham` incorrectly classified as `spam` (False Alarm).
  - **False Negative (FN):** Actual `spam` incorrectly classified as `ham` (Missed Spam).
  - **True Positive (TP):** Actual `spam` correctly classified as `spam`.

---

## Model Evaluation Results

Empirical results measured on 1,115 unseen test messages (`random_state=42`):

| Metric | Measured Test Result | Percentage |
|---|---:|---:|
| **Total Test Samples** | 1,115 messages | 100.00% |
| **Accuracy Score** | 0.9596 | **95.96%** |
| **Spam Precision** | 1.0000 | **100.00%** |
| **Spam Recall** | 0.6980 | **69.80%** |
| **Spam F1-Score** | 0.8221 | **82.21%** |

### Confusion Matrix Breakdown
- **True Negatives (TN):** 966 (Legitimate `ham` correctly classified)
- **False Positives (FP):** 0 (Zero false positives in this test split, yielding 100.00% precision on the evaluated 1,115 test samples. Note that 100.00% precision applies strictly to this test set split and does not guarantee zero false positives on future unseen data.)
- **False Negatives (FN):** 45 (Spam messages misclassified as `ham`)
- **True Positives (TP):** 104 (Spam messages correctly flagged)



---

## Installation & Setup (Windows PowerShell)

### Step 1: Clone Repository & Open PowerShell
Open Windows PowerShell and navigate to the project directory:
```powershell
cd "e:\Projects\Artificial Intelligence Internship\Project-1-Spam Email Classifier"
```

### Step 2: Create & Activate Virtual Environment
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```
*Note: If PowerShell blocks script execution due to execution policy, install packages directly into your environment using:*
```powershell
python -m pip install -r requirements.txt
```

### Step 3: Verify Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Dataset Setup Instructions

1. Download the official UCI zip archive from:
   [https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip](https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip)
2. Extract `sms+spam+collection.zip`.
3. Move the file named `SMSSpamCollection` (without extension) into the `data/` folder:
   `data/SMSSpamCollection`
4. Confirm path: `Project-1-Spam-Email-Classifier\data\SMSSpamCollection`.

---

## Running the Project

### 1. Train Model & Save Pipeline
Run the training script to load data, perform split, fit pipeline, display evaluation metrics, and save `models/spam_message_pipeline.joblib`:

```powershell
python src/train_model.py
```

### 2. Predict New Messages interactively
Run the interactive CLI predictor script to test custom text inputs:

```powershell
python src/predict_message.py
```

---

## Sample Test Messages

Test the model in `predict_message.py` using these 5 representative sample strings:

1. **Normal Personal Communication (`ham`):**
   > *"Hey Luffy, we are meeting at the Sunny deck for lunch at 12:30. Let me know if you can bring meat!"*
2. **Promotional Spam (`spam`):**
   > *"URGENT! You have won a FREE camera phone! Call 09061701461 right now to claim your prize. T&Cs apply."*
3. **Prize or Reward Scam (`spam`):**
   > *"CONGRATULATIONS! You have been selected to win 500,000 Berries cash reward! Text WIN to 88888 immediately to claim."*
4. **Short Ambiguous Message (`ham`):**
   > *"Ok call me back later."*
5. **One Piece-Inspired Fictional Test Message:**
   > *"SECRET MARINE ALERT: Bounty notice update! Click here to report Straw Hat Luffy for 3,000,000,000 Berries cash transfer."*

*Note: The One Piece-inspired test message is a user-created demonstration example and is not part of the UCI training dataset.*

---

## Presentation & Theme Note

This project uses a light, tasteful *One Piece* anime presentation theme (`Grand Line Marine Intelligence`, `Crew Message`, `Marine Alert`) in CLI log headers and user interface responses. This theme is purely for presentation and educational motivation; it does not alter standard machine learning algorithms or data processing workflows.

---

## Limitations

- **Text-Only Scope:** Designed for plain-text SMS strings, not raw emails with MIME headers or file attachments.
- **English Dataset Bounds:** Trained specifically on the English-language UCI dataset.
- **Dataset Age:** The UCI dataset reflects SMS spam patterns from the mid-2000s; modern phishing messages may exhibit different vocabulary patterns.
- **Probability Scores:** Output probability scores represent model probability estimates based on Naive Bayes term counts, not absolute certainty.
- **False Positives/Negatives:** Short or out-of-vocabulary messages may occasionally be misclassified.

---

## Future Improvements

- Incorporate email-specific datasets (e.g., Enron Spam Dataset or SpamAssassin public corpus).
- Add character and word n-grams (e.g., `ngram_range=(1, 2)`) to capture multi-word phrases.
- Compare `MultinomialNB` with `LogisticRegression` and `LinearSVC`.
- Add an error-analysis module to log misclassified messages for model auditing.
- Introduce an "uncertain" threshold category when estimated probabilities fall between 40% and 60%.

---

## GitHub & Internship Submission Checklist

- [x] Upload source code (`src/train_model.py`, `src/predict_message.py`), `requirements.txt`, `README.md`, `.gitignore`, `data/README.md`, and `reports/project_report.md`.
- [x] Ensure `.venv`, `__pycache__`, raw `SMSSpamCollection`, and binary `.joblib` files are listed in `.gitignore`.
- [x] Confirm repository contains no private environment files or credentials.
- [x] Include dataset credit and UCI reference links in documentation.

---

## License & Attribution

- **Dataset Credit:** UCI SMS Spam Collection created by Tiago A. Almeida and José María Gómez Hidalgo. Hosted on the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection).
- **Project Code:** Created as part of an Artificial Intelligence Internship project. Educational open-source use.

---

## Learning Outcomes

By completing this project, the following core machine learning competencies were demonstrated:
1. Understanding binary text classification concepts.
2. Building leakage-free ML pipelines using scikit-learn.
3. Feature engineering via TF-IDF vectorization.
4. Evaluating models using precision, recall, F1-score, and confusion matrices beyond simple accuracy.
5. Saving and loading trained model pipelines using `joblib`.
6. Constructing interactive command-line interfaces for real-time model inference.
