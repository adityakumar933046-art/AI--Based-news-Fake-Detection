
import pandas as pd
import re
import nltk
import joblib

from pathlib import Path
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Download stopwords if required
nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "dataset" / "cleaned_welfake.csv"
MODEL_DIR = BASE_DIR / "model"

# Load cleaned dataset
df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# Validate required columns
required_columns = {"content", "label"}

if not required_columns.issubset(df.columns):
    raise ValueError("Dataset must contain content and label columns.")

# Remove invalid records
df = df.dropna(subset=["content", "label"])

# Check labels
if set(df["label"].unique()) != {0, 1}:
    raise ValueError("Expected labels 0 (Fake) and 1 (Real).")

# Text preprocessing
def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\S+|www\S+", "", text)

    text = re.sub(r"[^a-z\s]", "", text)

    words = text.split()

    words = [word for word in words if word not in stop_words]

    return " ".join(words)


df["content"] = df["content"].apply(clean_text)

# Remove empty articles after cleaning
df = df[df["content"].str.strip() != ""]

# Input and target
X = df["content"]
y = df["label"].astype(int)

print("\nLabel distribution:")
print(y.value_counts())

# Split dataset: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# TF-IDF vectorization
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

print("\nTF-IDF vectorization completed!")

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB()
}

results = {}

# Train and evaluate models
for name, model in models.items():

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)

    results[name] = accuracy

    print("\n" + "=" * 50)
    print("Model:", name)
    print("Accuracy:", round(accuracy * 100, 2), "%")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            prediction,
            labels=[0, 1],
            target_names=["FAKE", "REAL"],
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, prediction, labels=[0, 1]))


# Select best model based on test accuracy
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

# Save trained model and vectorizer
MODEL_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(
    best_model,
    MODEL_DIR / "fake_news_model.pkl"
)

joblib.dump(
    vectorizer,
    MODEL_DIR / "tfidf_vectorizer.pkl"
)

print("\n" + "=" * 50)
print("Best Model:", best_model_name)
print("Best Accuracy:", round(results[best_model_name] * 100, 2), "%")
print("Model saved successfully!")
