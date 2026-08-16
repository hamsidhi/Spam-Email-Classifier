# Grand Line Message Bounty Detector — Technical Internship Project Report

**Project Title:** Grand Line Message Bounty Detector — SMS Spam Message Classifier  
**Author:** AI Intern  
**Domain:** Artificial Intelligence / Natural Language Processing  
**Dataset:** UCI Machine Learning Repository — SMS Spam Collection  
**Execution Environment:** Windows 11 / PowerShell / VS Code  
**Date:** August 2026  

---

## 1. Abstract

This report documents the design, implementation, and empirical evaluation of the **Grand Line Message Bounty Detector**, a lightweight Natural Language Processing (NLP) machine learning application built to classify text messages into `ham` (legitimate communication) or `spam` (unwanted or fraudulent notices). The downloaded UCI SMS Spam Collection source file contains 5,574 physical lines. After parsing the expected tab-separated `label` and `message` fields, the training script loaded 5,572 valid records: 4,825 `ham` messages and 747 `spam` messages. Operating on these 5,572 parsed records, the system implements an end-to-end scikit-learn pipeline combining Term Frequency-Inverse Document Frequency (`TfidfVectorizer`) text feature extraction with a probabilistic `MultinomialNB` classifier. Fitted strictly on an 80% training split (`random_state=42`), the system achieved an overall test accuracy of **95.96%**, a test set spam precision score of **100.00%** (0 false positives on the test split), a spam recall of **69.80%**, and a spam F1-score of **82.21%** on 1,115 unseen test messages.

---

## 2. Introduction

Unwanted text communications disrupt daily user experiences and present serious security risks through social engineering, phishing, and financial scams. In machine learning internships, text classification serves as a fundamental benchmark problem. This project constructs an understandable, reproducible, and baseline machine learning pipeline for text classification without relying on complex deep learning architectures, heavy web frameworks, APIs, or external database systems.

---

## 3. Problem Statement

Modern messaging environments are saturated with automated promotional messages and financial scam notices. The primary technical challenge is automatically recognizing subtle textual patterns that distinguish legitimate messages from spam, while guaranteeing that legitimate user messages are never falsely flagged as spam (minimizing false positives).

---

## 4. Project Objectives

1. Develop a clean, end-to-end Python NLP baseline classifier.
2. Ensure strict data leakage prevention by vectorizing features inside a scikit-learn pipeline fitted only on training data.
3. Quantify performance using Accuracy, Precision, Recall, F1-score, and Confusion Matrix analysis.
4. Save the trained pipeline using `joblib` for instant real-time inference.
5. Provide an interactive command-line interface for testing single messages with estimated class probabilities.

---

## 5. Project Scope

This project focuses on binary text classification for plain-text short messages. It includes data loading, data validation, stratified splitting, feature weighting, probabilistic model training, comprehensive metric reporting, model serialization, and interactive prediction.

---

## 6. SMS Versus Email Classification Clarification

- **SMS Plain Text Scope:** The UCI dataset contains plain-text SMS messages. Plain SMS text lacks MIME headers, HTML structures, routing paths, or attachments.
- **Email Security System Distinction:** A production-grade email spam system evaluates multi-part MIME headers (`From:`, `Return-Path:`, `DKIM-Signature`), HTML layout tags, embedded URLs, domain reputation, network IPs, and attachment signatures.
- **Technical Context:** While titled *"Spam Email Classifier"* in the internship curriculum, this project implements the core Natural Language Processing engine responsible for text body analysis.

---

## 7. Dataset Description

The project uses the **UCI SMS Spam Collection**:
- **Source File & Parsing:** The downloaded source file contains 5,574 physical lines. After parsing the expected tab-separated `label` and `message` fields, the training script loaded 5,572 valid records: 4,825 `ham` messages and 747 `spam` messages. The experiment and all reported metrics are based on these 5,572 parsed records.
- **Language:** English.
- **Structure:** Tab-separated raw text (`label\tmessage`).
- **Class Labels:** `ham` (legitimate) and `spam` (unwanted/fraudulent).
- **Distribution:** 4,825 `ham` messages (86.59%) and 747 `spam` messages (13.41%).

---

## 8. Dataset Attribution

The dataset was compiled and made available by **Tiago A. Almeida** and **José María Gómez Hidalgo**. Hosted on the official [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection).

---

## 9. Data Dictionary

| Column Name | Data Type | Description | Values |
|---|---|---|---|
| `label` | String | Target classification label | `ham`, `spam` |
| `message` | String | Raw text string of the message | Free-form text |

---

## 10. Software and Hardware Requirements

- **Operating System:** Windows 10/11 (PowerShell).
- **Programming Language:** Python 3.8+.
- **Libraries:** pandas (>= 2.0.0), scikit-learn (>= 1.2.0), joblib (>= 1.3.0).
- **Hardware:** Standard CPU (No GPU required).

---

## 11. Project Structure

```
Project-1-Spam-Email-Classifier/
├── data/
│   ├── SMSSpamCollection
│   └── README.md
├── models/
│   ├── .gitkeep
│   └── spam_message_pipeline.joblib
├── reports/
│   └── project_report.md
├── src/
│   ├── train_model.py
│   └── predict_message.py
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 12. System Architecture

```
SMSSpamCollection (Tab-separated data)
        ↓
Data Loading & Validation (src/train_model.py)
        ↓
Stratified Train/Test Split (80% train / 20% test, random_state=42)
        ↓
scikit-learn Pipeline Construction
   ├── TfidfVectorizer (lowercase=True, strip_accents='unicode', sublinear_tf=True)
   └── MultinomialNB Classifier
        ↓
Pipeline Training (Fitted on train set only)
        ↓
Evaluation on Test Data (X_test, y_test)
        ↓
Model Serialization (models/spam_message_pipeline.joblib)
        ↓
Interactive Predictor CLI (src/predict_message.py)
```

---

## 13. Data Flow

1. **Input:** Text string received.
2. **Transformation:** Lowercased, accents stripped, and transformed into a numerical sparse TF-IDF feature vector.
3. **Classification:** Multinomial Naive Bayes computes log probability scores for `ham` vs `spam`.
4. **Output:** Highest probability class assigned with estimated percentage.

---

## 14. Data Validation and Preprocessing

- **Missing Data:** Inspected via `df.isnull().sum()`. 0 missing values detected.
- **Label Validation:** Confirmed that raw dataset contains strictly `ham` and `spam`.
- **Text Cleaning:** Handled automatically inside `TfidfVectorizer` (lowercase conversion, accent stripping, tokenization).

---

## 15. Train/Test Split

- **Ratio:** 80% Training (4,457 records), 20% Testing (1,115 records).
- **Method:** `train_test_split(..., test_size=0.20, random_state=42, stratify=y)`.
- **Rationale:** Stratification ensures exact balance maintenance (~86.59% `ham` / 13.41% `spam`) across both splits.

---

## 16. TF-IDF Vectorization

Term Frequency-Inverse Document Frequency (TF-IDF) converts raw text into numerical features:
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$
- **`sublinear_tf=True`:** Replaces TF with $1 + \log(\text{TF})$ to prevent high term frequency dominance.
- **`strip_accents='unicode'`:** Standardizes special characters.

---

## 17. Multinomial Naive Bayes

Multinomial Naive Bayes estimates posterior probabilities using Bayes' Theorem under feature independence assumptions:
$$P(y \mid x_1, \dots, x_n) \propto P(y) \prod_{i=1}^{n} P(x_i \mid y)$$
It is mathematically optimal for sparse text count data and fast to train on CPU.

---

## 18. Training Process

Training executed via `src/train_model.py`:
1. Vectorizer fitted on `X_train` vocabulary.
2. Classifier fitted on transformed `X_train` vectors and `y_train` target labels.
3. Unified pipeline saved to `models/spam_message_pipeline.joblib`.

---

## 19. Evaluation Metrics Formulae

- **Accuracy:** $\frac{TP + TN}{TP + TN + FP + FN}$
- **Precision (`spam`):** $\frac{TP}{TP + FP}$
- **Recall (`spam`):** $\frac{TP}{TP + FN}$
- **F1-Score (`spam`):** $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$

---

## 20. Actual Experimental Results

Experimental run on 1,115 unseen test samples (`random_state=42`):

| Metric | Measured Value | Percentage |
|---|---:|---:|
| **Total Test Records** | 1,115 | 100.00% |
| **Accuracy Score** | 0.9596 | 95.96% |
| **Spam Precision** | 1.0000 | 100.00% |
| **Spam Recall** | 0.6980 | 69.80% |
| **Spam F1-Score** | 0.8221 | 82.21% |

### Classification Report Output:

```text
              precision    recall  f1-score   support

         ham       0.96      1.00      0.98       966
        spam       1.00      0.70      0.82       149

    accuracy                           0.96      1115
   macro avg       0.98      0.85      0.90      1115
weighted avg       0.96      0.96      0.96      1115
```

---

## 21. Confusion Matrix Analysis

```text
                 Predicted 'ham'   Predicted 'spam'
Actual 'ham'  :       966             0
Actual 'spam' :       45              104
```

- **True Negatives (TN = 966):** 966 legitimate messages correctly identified as `ham`.
- **False Positives (FP = 0):** 0 legitimate messages falsely flagged as `spam` on this test split (yielding 100.00% precision on the evaluated 1,115 test samples). Note that 100.00% precision applies strictly to this test set split and does not guarantee zero false positives on future unseen data.
- **False Negatives (FN = 45):** 45 spam messages misclassified as `ham` (due to conservative Naive Bayes decision boundary).
- **True Positives (TP = 104):** 104 spam messages correctly flagged as `spam`.

---

## 22. Interactive Sample Predictions

Tested via `src/predict_message.py`:

| Input Message | Predicted Label | Estimated Probability | Note / Type |
|---|---|---:|---|
| *"Hey, are we still meeting at 6 pm today?"* | `ham` | 99.71% | Standard personal text |
| *"Congratulations! You have won a cash prize. Call now to claim your reward."* | `spam` | 91.76% | Promotional prize scam |
| *"Urgent! Claim your free treasure reward by sending your details immediately."* | `spam` | 66.21% | Fraudulent spam notice |
| *"Can you call me later?"* | `ham` | 99.77% | Short legitimate question |
| *"Congratulations, Straw Hat crew! You have won 1,000,000 berries. Click now to claim your bounty."* | `spam` | 78.52% | *Fictional One Piece test message* |

*Note: The One Piece-themed test message is a user-created demonstration example and is not part of the UCI dataset.*

---

## 23. Error Analysis Template & Review

- **Zero False Positives in Test Split:** The model achieved 100.00% precision on the 1,115 test samples, resulting in 0 false alarms on the test set.
- **False Negative Count (45 samples):** The current training script calculates and reports the aggregate count of false negatives (45 samples) via the confusion matrix, but it does not output or inspect the raw text strings of those misclassified messages.
- **Template for Future Error Analysis:** To conduct qualitative error analysis in future iterations, an error analysis script can filter `X_test[(y_test == "spam") & (y_pred == "ham")]` to output the exact text of the 45 misclassified spam messages for manual review.

---

## 24. Limitations

1. Plain SMS text scope; lacks email headers or HTML parsing capabilities.
2. English-only vocabulary.
3. Static dataset features reflecting mid-2000s SMS spam patterns.
4. Model probabilities represent Naive Bayes estimated likelihoods, not absolute certainty.

---

## 25. Ethical Considerations

- **User Privacy:** Message text classification must respect content privacy.
- **False Positive Impact:** Flagging legitimate messages as spam can cause loss of critical user communication.

---

## 26. Future Improvements

1. Extend training to email corpora (Enron / SpamAssassin).
2. Incorporate word and character n-grams (`ngram_range=(1, 2)`).
3. Compare MultinomialNB with LogisticRegression and LinearSVC.
4. Implement a decision threshold tuning module to improve spam recall without increasing false positives.

---

## 27. Conclusion

The **Grand Line Message Bounty Detector** successfully establishes an understandable, reproducible, and highly precise baseline model for SMS spam text classification. By achieving 95.96% accuracy and 100.00% test set precision on unseen test data, the pipeline satisfies all internship requirements while preventing data leakage.

---

## 28. Learning Outcomes

- End-to-end NLP data processing and validation in pandas.
- Data leakage prevention using scikit-learn Pipelines.
- Feature engineering with TF-IDF vectorization.
- Evaluation using precision, recall, F1-score, and confusion matrices.
- Pipeline serialization using `joblib`.
- Interactive CLI development for real-time model inference.

---

## 29. References

1. **UCI Machine Learning Repository:** SMS Spam Collection Dataset.  
   URL: [https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
2. **Dataset Authors:** Tiago A. Almeida and José María Gómez Hidalgo.
3. **Scikit-Learn Documentation:** Working with Text Data & Pipelines.  
   URL: [https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)

---

## 30. One Piece Theme Code Mapping & Terminology Dictionary

To ensure complete clarity and accessibility for code reviewers and recruiters, the table below documents the exact mapping between the *One Piece* themed identifiers used in the Python source code (`src/train_model.py` & `src/predict_message.py`) and standard machine learning / Python terminology:

| Themed Code Identifier | Standard Technical / ML Concept | Technical Explanation & Rationale |
|---|---|---|
| `grand_line_root` | Project Root Directory (`Path`) | Resolves the root workspace folder relative to the script location. |
| `bounty_dataset_path` | Dataset File Path (`SMSSpamCollection`) | Points to `data/SMSSpamCollection` tab-separated text dataset. |
| `inspect_bounty_scrolls()` | `load_and_validate_data()` | Function that reads tab-separated data, checks nulls, and prints class balance. |
| `scrolls_df` | `pandas.DataFrame` | The tabular DataFrame holding `label` and `message` columns. |
| `grand_line_messages` | `X` (Feature Series) | pandas Series containing raw text messages. |
| `bounty_labels` | `y` (Target Series) | pandas Series containing `ham` or `spam` classification targets. |
| `crew_train_messages`, `crew_test_messages` | `X_train`, `X_test` | The 80% training and 20% testing message feature splits. |
| `crew_train_labels`, `crew_test_labels` | `y_train`, `y_test` | The 80% training and 20% testing target label splits. |
| `devil_fruit_vectorizer` | `TfidfVectorizer` | TF-IDF transformer converting raw text into numerical feature vectors. |
| `marine_classifier` | `MultinomialNB` | Probabilistic Naive Bayes classifier estimating posterior class probabilities. |
| `marine_spam_pipeline` | `sklearn.pipeline.Pipeline` | Unified scikit-learn pipeline coupling vectorizer and classifier. |
| `train_marine_fleet_pipeline()` | `train_pipeline()` | Function constructing and fitting the pipeline strictly on training data. |
| `evaluate_battle_results()` | `evaluate_model()` | Function computing Accuracy, Precision, Recall, F1, and Confusion Matrix. |
| `save_marine_den_den_mushi()` | `joblib.dump()` | Serializes the trained pipeline artifact to `models/spam_message_pipeline.joblib`. |
| `load_marine_pipeline()` | `joblib.load()` | Loads serialized pipeline binary for interactive inference. |
| `inspect_message()` | `predict_message()` | Function predicting label and probability for new text input strings. |
| `[CREW MESSAGE]` | Class `ham` (Legitimate) | Presentation tag for legitimate non-spam messages. |
| `[MARINE ALERT]` | Class `spam` (Spam/Scam) | Presentation tag for unwanted spam or fraudulent notices. |

