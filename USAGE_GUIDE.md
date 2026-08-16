# 🏴‍☠️ Grand Line Message Bounty Detector — Step-by-Step User Guide

Welcome, Navigator! This step-by-step guide provides complete instructions for setting up, training, testing, and running the **Grand Line Message Bounty Detector — SMS Spam Message Classifier**.

---

## 📋 Prerequisites & Tools
Before getting started, ensure you have:
* **Python 3.8+** installed on your system.
* **Windows PowerShell** or standard terminal shell.
* **VS Code** (recommended IDE).
* **Web Browser** (Chrome, Edge, Firefox, or Brave) for the web application.

---

## 📥 Step 1: Clone Repository & Set Up Virtual Environment

Open **Windows PowerShell** in VS Code or your terminal:

```powershell
# Navigate to the project root directory
cd "e:\Projects\Artificial Intelligence Internship\Project-1-Spam Email Classifier"

# Create a virtual environment named .venv
python -m venv .venv

# Activate the virtual environment (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Upgrade pip and install required lightweight dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> **Note for Execution Policy Issues:**  
> If PowerShell blocks script activation (`Activate.ps1`), you can bypass execution policies for the session using:  
> `Set-ExecutionPolicy -ExecutionPolicy Unrestricted -Scope Process`

---

## 📂 Step 2: Download & Place the Dataset

1. Download the official UCI SMS Spam Collection archive from:  
   👉 [https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip](https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip)
2. Extract the zip file contents.
3. Locate the raw dataset file named **`SMSSpamCollection`** (no file extension).
4. Move or copy this file into the `data/` folder of your project:  
   **Target Path:** `data/SMSSpamCollection`

---

## ⚔️ Step 3: Train Model & Generate Pipeline Artifact

Execute the One Piece-themed training script to load data, perform stratified 80/20 train/test splitting, train the TF-IDF + Naive Bayes pipeline, display battle metrics, and serialize the trained model artifact:

```powershell
python src/train_model.py
```

### Expected Output Summary:
* **Total Scrolls Loaded:** 5,572 valid records (4,825 `ham` / 747 `spam`).
* **Train Split:** 4,457 messages (80%).
* **Test Split:** 1,115 messages (20%).
* **Measured Battle Metrics:**
  * **Accuracy:** 95.96%
  * **Spam Precision:** 100.00% (0 False Positives)
  * **Spam Recall:** 69.80%
  * **Spam F1-Score:** 82.21%
* **Saved Artifact:** `models/spam_message_pipeline.joblib`

---

## 🐚 Step 4: Run the Interactive Terminal Inspector CLI

To test single text messages or custom prompts in real time via command-line:

```powershell
python src/predict_message.py
```

### Interactive Usage Instructions:
1. Type or paste any text message prompt when asked `GrandLine-Inspector>`.
2. Press **Enter** to scan the message.
3. **Read the Output:**
   * **`[CREW MESSAGE]`**: Safe message from Straw Hat crewmates (`ham`).
   * **`[MARINE ALERT]`**: Unwanted pirate spam or fraudulent notice (`spam`).
   * **Model Probability:** Estimated percentage confidence.
4. Type `exit`, `quit`, or `q` to return to your ship.

---

## 🌐 Step 5: Launch the Interactive Web Frontend Application

You have two easy options to run the **One Piece Web Application**:

### Option A: Direct File Launch (Quickest)
1. Open File Explorer and navigate to `frontend/`.
2. Double-click **`index.html`** to open the web app instantly in your browser!

### Option B: Local Development Server (Recommended)
Run the built-in Python web server from your PowerShell terminal:

```powershell
python -m http.server 8000 --directory frontend
```

Now open your web browser and navigate to:  
👉 **`http://localhost:8000`**

### Features in the Web App:
* **Live Bounty Inspector Tab:** Paste messages or click sample chips (**Crewmate**, **Promo Scam**, **Bounty Scam**, **One Piece Alert**) to dynamically render Wanted Bounty Posters!
* **Model Performance & Metrics Tab:** View stat cards for Accuracy, Precision, Recall, F1-Score, and the interactive Confusion Matrix breakdown.
* **One Piece Code Dictionary Tab:** Inspect the full mapping reference guide connecting themed identifiers (`devil_fruit_vectorizer`, `marine_classifier`) to standard ML concepts.

---

## 📜 Step 6: Internship Technical Report & Code Dictionary

* **Technical Report File:** `reports/project_report.md`  
  Contains 30 formal sections including abstract, methodology, TF-IDF formulas, Naive Bayes Bayes' Theorem, dataset details, confusion matrix analysis, and ethical disclosures.
* **Code Dictionary:** Section 30 of `reports/project_report.md` provides a lookup dictionary explaining every themed variable name.

---

## 🐙 Step 7: Push Everything to GitHub

To ensure your GitHub repository contains all code, documentation, web frontend files, and user guides:

```powershell
# 1. Check current git status
git status

# 2. Stage all project files
git add .

# 3. Commit with a descriptive message
git commit -m "Complete One Piece Spam Classifier with trained pipeline, reports, web frontend, and user guide"

# 4. Push changes to GitHub main branch
git push origin main
```

---

## ⚓ Summary of Repository File Map

```text
Project-1-Spam-Email-Classifier/
├── 📂 data/
│   ├── SMSSpamCollection           # Raw dataset file (git-ignored)
│   └── README.md                   # Dataset setup instructions
├── 📂 frontend/
│   ├── index.html                  # Web app UI layout
│   ├── style.css                   # Dark ocean glassmorphic styling
│   ├── app.js                      # Client-side NLP engine & tab handlers
│   └── 📂 assets/
│       └── header.png              # One Piece header banner artwork
├── 📂 models/
│   ├── .gitkeep                    # Directory structure tracking
│   └── spam_message_pipeline.joblib # Serialized trained pipeline
├── 📂 reports/
│   └── project_report.md           # Formal 30-section technical internship report
├── 📂 src/
│   ├── train_model.py              # Model training script
│   └── predict_message.py          # Interactive terminal CLI inspector
├── .gitignore                      # Git exclusion rules
├── README.md                       # Main repository README
├── USAGE_GUIDE.md                  # Detailed step-by-step user guide (this file)
└── requirements.txt                # Lightweight dependencies
```

*May fair winds guide your machine learning journey across the Grand Line! 🏴‍☠️*
