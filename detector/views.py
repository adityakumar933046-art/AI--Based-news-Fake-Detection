
from django.shortcuts import render

from .models import NewsPrediction
from .services.google_fact_check import search_fact_checks

import re
import joblib

from pathlib import Path
from nltk.corpus import stopwords


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "model"

MODEL_PATH = MODEL_DIR / "fake_news_model.pkl"

VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"


model = None
vectorizer = None

if MODEL_PATH.exists() and VECTORIZER_PATH.exists():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)


try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    stop_words = set()


def clean_text(text):

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)

    words = text.split()
    words = [word for word in words if word not in stop_words]

    return " ".join(words)


def home(request):
    return render(request, "detector/home.html")


def predict_news(request):

    if request.method == "POST":

        news_text = request.POST.get("news", "").strip()

        if not news_text:
            return render(request, "detector/home.html", {
                "error": "Please enter news text."
            })

        if model is None or vectorizer is None:
            return render(request, "detector/home.html", {
                "error": "ML model not found. Train the model first."
            })

        cleaned_news = clean_text(news_text)

        if not cleaned_news.strip():
            return render(request, "detector/home.html", {
                "error": "Please enter meaningful news text."
            })

        news_vector = vectorizer.transform([cleaned_news])

        prediction = model.predict(news_vector)[0]

        probabilities = model.predict_proba(news_vector)[0]

        confidence = max(probabilities) * 100

        result = "FAKE" if prediction == 0 else "REAL"

        # Search Google's existing fact-check database.
        # This is a text search, not independent verification.
        fact_check = search_fact_checks(news_text[:500])

        fact_check_results = fact_check.get("results", [])

        if not fact_check.get("success"):
            fact_check_status = "API_ERROR"
            fact_check_error = fact_check.get("error", "Unknown API error.")

        elif fact_check_results:
            fact_check_status = "MATCHES_FOUND"
            fact_check_error = ""

        else:
            fact_check_status = "NO_MATCHES"
            fact_check_error = ""

        # Save the complete analysis in the database.
        NewsPrediction.objects.create(
            news_text=news_text,
            prediction=result,
            confidence=confidence,
            fact_check_results=fact_check_results,
            fact_check_status=fact_check_status,
            fact_check_error=fact_check_error
        )

        return render(request, "detector/result.html", {
            "news": news_text,
            "result": result,
            "confidence": round(confidence, 2),
            "fact_check_results": fact_check_results,
            "fact_check_status": fact_check_status,
            "fact_check_error": fact_check_error
        })

    return render(request, "detector/home.html")


def history(request):

    predictions = NewsPrediction.objects.all().order_by("-created_at")

    return render(request, "detector/history.html", {
        "predictions": predictions
    })
