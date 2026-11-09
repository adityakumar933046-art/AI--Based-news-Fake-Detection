import pandas as pd
import numpy as np
import re
import nltk
import joblib
import os

from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

fake = pd.read_csv(os.path.join(BASE_DIR, 'dataset', 'Fake.csv'))
real = pd.read_csv(os.path.join(BASE_DIR, 'dataset', 'True.csv'))

fake['label'] = 0
real['label'] = 1

df = pd.concat([fake, real], ignore_index=True)

df = df.dropna(subset=['text'])

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r'http\S+|www\S+', '', text)

    text = re.sub(r'[^a-z\s]', '', text)

    words = text.split()

    words = [word for word in words if word not in stop_words]

    return ' '.join(words)


df['text'] = df['text'].apply(clean_text)

X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_train = vectorizer.fit_transform(X_train)

X_test = vectorizer.transform(X_test)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Naive Bayes': MultinomialNB()
}

results = {}

for name, model in models.items():

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)

    results[name] = accuracy

    print("\nModel:", name)

    print("Accuracy:", accuracy)

    print(classification_report(y_test, prediction))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, prediction))


best_model_name = max(results, key=results.get)

best_model = models[best_model_name]

MODEL_DIR = os.path.join(BASE_DIR, 'model')

os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(
    best_model,
    os.path.join(MODEL_DIR, 'fake_news_model.pkl')
)

joblib.dump(
    vectorizer,
    os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
)

print("\nBest Model:", best_model_name)

print("Model saved successfully!")
