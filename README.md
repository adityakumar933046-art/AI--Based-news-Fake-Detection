# AI-Based Fake News Detection System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-green.svg)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A complete, dynamic, and fully functional **AI-Based Fake News Detection System** designed as a college Artificial Intelligence and Machine Learning project. The system analyzes raw news headlines and articles, extracts statistical unigram/bigram TF-IDF features using NLP with negation-preservation, and classifies authenticity using an ensemble of **two machine learning algorithms**: **Logistic Regression** and **Multinomial Naive Bayes**.

---

## 1. System Architecture

```text
USER
  ↓
WEB INTERFACE (HTML, CSS, JavaScript)
  ↓
FLASK BACKEND (`app.py`)
  ↓
NLP PREPROCESSING (`src/preprocessing.py`)
  ↓
TF-IDF FEATURE EXTRACTION (`models/tfidf_vectorizer.pkl`)
  ↓
LOGISTIC REGRESSION + MULTINOMIAL NAIVE BAYES
  ↓
COMPARE PREDICTIONS
  ↓
REAL / FAKE / UNCERTAIN
  ↓
SQLITE DATABASE (`database/database.db`) + DYNAMIC DASHBOARD (`/dashboard`)
```

### Prediction Consensus Rules
- **Both models predict REAL** $\rightarrow$ **REAL**
- **Both models predict FAKE** $\rightarrow$ **FAKE**
- **Models disagree** $\rightarrow$ **UNCERTAIN**

---

## 2. Technologies Used

| Domain | Technology / Library | Role & Functionality |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Core language for machine learning, pipeline, and backend |
| **Web Framework** | Flask 3.x | WSGI web framework handling REST endpoints and page rendering |
| **Data Manipulation** | Pandas & NumPy | High-performance tabular data extraction, deduplication, and matrix operations |
| **NLP & Text Processing**| NLTK | Stopwords processing with specialized negation preservation |
| **Feature Extraction** | Scikit-learn (TF-IDF) | Sublinear TF-IDF (unigram + bigram, 8,000 max features) |
| **Machine Learning** | Logistic Regression | Probabilistic linear classification with L2 regularization |
| **Machine Learning** | Multinomial Naive Bayes | Probabilistic conditional frequency classifier with Laplace smoothing |
| **Decision Logic** | Two-Model Consensus | Compares both model predictions: Real, Fake, or Uncertain |
| **Database** | SQLite 3 | Embedded relational store for historical predictions and query audit log |
| **Frontend UI** | HTML5, CSS3, JavaScript | Minimalist, responsive two-color UI (`#14213D` & `#FFFFFF`) |
| **Visual Charts** | Matplotlib & Chart.js | Exported publication figures (PNG) and real-time interactive JavaScript graphs |

---

## 3. Machine Learning Algorithms

### 1. Logistic Regression
- Models the log-odds of news authenticity using an L2-regularized logistic function:
  $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
- Provides calibrated prediction probabilities via `predict_proba`.

### 2. Multinomial Naive Bayes
- Computes posterior class probabilities based on word frequencies across the TF-IDF feature space:
  $$P(C_k \mid \mathbf{x}) \propto P(C_k) \prod_{i=1}^{n} P(x_i \mid C_k)$$
- Fast, stable, and highly effective for high-dimensional sparse text vectors with Laplace smoothing ($\alpha = 0.1$).

---

## 4. Project Directory Structure

```text
AI-FAKE-NEWS-DETECTION/
│
├── app.py                      # Flask main entrypoint & web routes
├── train_model.py              # ML training, evaluation, and chart generation script
├── requirements.txt            # Python dependencies
├── README.md                   # Full college project documentation
├── .gitignore                  # Git ignore rules for bytecode, DB, and datasets
│
├── dataset/                    # Dataset directory
│   ├── news_dataset.csv        # Unified, deduplicated news dataset (content, label)
│   ├── Fake.csv                # Raw fake news records
│   ├── True.csv                # Raw genuine news records
│   └── .gitkeep
│
├── models/                     # Serialized models and evaluation artifacts
│   ├── logistic_regression.pkl # Trained Logistic Regression model
│   ├── naive_bayes.pkl         # Trained Multinomial Naive Bayes model
│   ├── tfidf_vectorizer.pkl    # Fitted 8,000-term TF-IDF Vectorizer
│   ├── evaluation_metrics.json # Real empirical test evaluation metrics
│   └── .gitkeep
│
├── src/                        # Core application modules
│   ├── __init__.py
│   ├── preprocessing.py        # Text normalization & negation-preserving cleaner
│   ├── prediction.py           # Two-model prediction & consensus service
│   ├── database.py             # SQLite database management, queries, and migrations
│   └── evaluation.py           # Metric calculation and Matplotlib plot generator
│
├── database/                   # SQLite storage directory
│   ├── database.db             # SQLite database instance (auto-created on startup)
│   └── .gitkeep
│
├── templates/                  # Jinja2 HTML templates
│   ├── base.html               # Shared layout, navigation, and footer
│   ├── index.html              # Home page: news input, test samples, result card
│   ├── dashboard.html          # Dynamic dashboard with Chart.js and KPI widgets
│   ├── history.html            # Query history with search, filter, and clear modal
│   └── about.html              # System overview and technology stack
│
├── static/                     # Web assets (Strict Two-Color Theme: #14213D & #FFFFFF)
│   ├── css/
│   │   └── style.css           # Minimalist responsive design stylesheet
│   ├── js/
│   │   └── script.js           # AJAX prediction handler, text counter, Chart.js logic
│   └── images/                 # Generated evaluation plots
│       ├── model_comparison.png# Matplotlib multi-metric comparison bar chart
│       └── confusion_matrices.png # Matplotlib test set confusion matrices
│
└── tests/                      # Automated unit and integration test suite
    └── test_app.py             # End-to-end tests for NLP, models, DB, and routes
```

---

## 5. Dataset Instructions

The project uses the standard **ISOT / Kaggle Fake and Real News Dataset** containing over 44,000 articles.

### Obtaining the Dataset
If cloning this repository on a new machine without dataset CSVs:
1. Download from Kaggle: [Fake and Real News Dataset by Clément Bisaillon](https://www.kaggle.com/datasets/clermont/fake-and-real-news-dataset)
2. Extract the archive to obtain `Fake.csv` and `True.csv`.
3. Place `Fake.csv` and `True.csv` inside `dataset/`.
4. Run `python train_model.py`. The script will automatically clean, deduplicate, and prepare `dataset/news_dataset.csv`.

---

## 6. Installation & Quick Start

```bash
# 1. Clone repository
git clone https://github.com/adityakumar933046-art/AI--Based-news-Fake-Detection.git
cd AI--Based-news-Fake-Detection

# 2. Create virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install requirements
pip install -r requirements.txt

# 4. Train the 2 models (generates metrics & models)
python train_model.py

# 5. Run automated tests
python -m unittest tests/test_app.py

# 6. Start the web application
python app.py
```

Open your browser at: **http://127.0.0.1:5000**

---

## 7. Empirical Model Evaluation

Evaluated on an independent, unseen test set of **3,200 samples** with stratified 50/50 class balance:

| Model | Accuracy | Precision | Recall | F1 Score | Training Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **98.72%** | **98.27%** | **99.19%** | **98.72%** | 0.14 s |
| **Multinomial Naive Bayes** | **95.94%** | **95.88%** | **96.00%** | **95.94%** | 0.01 s |

> **Note on Metric Authenticity**: Metrics are never hardcoded. Values are dynamically loaded from `models/evaluation_metrics.json` directly computed by scikit-learn.

---

## 8. Application Features

1. **Interactive News Analyzer**:
   - Clean textarea supporting long news articles and short headlines.
   - Live character and word counter.
   - 2 demonstration sample buttons ("Load Sample Real News", "Load Sample Fake News").
   - Non-blocking AJAX request with animated loading spinner.
2. **Transparent Consensus Breakdown**:
   - Outcome badge (**REAL NEWS**, **FAKE NEWS**, or **UNCERTAIN**).
   - Consensus summary (e.g. `Both models agree: REAL` or `Models disagree`).
   - Individual model cards showing calculated confidence percentages and progress bars.
   - Analysis latency timer (in milliseconds).
3. **Dynamic Analytics Dashboard**:
   - Real-time KPI widgets: Total Analyzed, Real Predictions, Fake Predictions, and Uncertain Predictions.
   - Two-color Chart.js bar chart showing empirical accuracy, precision, recall, and F1 scores.
   - Benchmark evaluation summary table.
4. **Comprehensive Prediction History**:
   - Tabular record of past queries with timestamps and model prediction splits.
   - Keyword search and prediction outcome filtering.
   - Clear history button with confirmation dialog.

---

## 9. Limitations & Ethical Considerations

- **Pattern Recognition vs. Fact-Checking**: This application analyzes statistical word associations, linguistic structure, and stylistic markers learned from training data. It does not replace human investigative journalism.
- **Temporal News Drift**: Language and subject matter evolve over time; retraining on newer news corpora is recommended periodically.
- **Disclaimer**: Every prediction screen presents a disclaimer reminding users that AI-based classification is an assistive tool and does not represent official journalistic verification.

---

## 10. License

This project is licensed under the MIT License - feel free to use and adapt it for academic, research, and educational purposes.
