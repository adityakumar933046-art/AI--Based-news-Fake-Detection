"""
Prediction and Consensus Service for AI Fake News Detection System.

Loads trained TF-IDF vectorizer and 2 machine learning models:
  1. Logistic Regression
  2. Multinomial Naive Bayes

Prediction Rules:
  - If both models predict REAL -> REAL
  - If both models predict FAKE -> FAKE
  - If models disagree -> UNCERTAIN
"""

import os
import sys
import time
import joblib
from typing import Dict, Any, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import clean_text
from src.database import insert_prediction


class FakeNewsPredictor:
    """
    Two-Model Fake News Predictor using Logistic Regression and Multinomial Naive Bayes.
    """

    def __init__(self, models_dir: Optional[str] = None):
        if models_dir is None:
            models_dir = os.path.join(PROJECT_ROOT, 'models')
        self.models_dir = models_dir
        self.vectorizer = None
        self.model_lr = None
        self.model_nb = None
        self.is_loaded = False
        self.load_models()

    def load_models(self):
        """Load vectorizer and the 2 trained models from disk."""
        vec_path = os.path.join(self.models_dir, 'tfidf_vectorizer.pkl')
        lr_path = os.path.join(self.models_dir, 'logistic_regression.pkl')
        nb_path = os.path.join(self.models_dir, 'naive_bayes.pkl')

        required_files = [vec_path, lr_path, nb_path]
        missing = [f for f in required_files if not os.path.exists(f)]
        if missing:
            raise FileNotFoundError(
                f"Missing model files: {missing}. Please run `python train_model.py` first."
            )

        self.vectorizer = joblib.load(vec_path)
        self.model_lr = joblib.load(lr_path)
        self.model_nb = joblib.load(nb_path)
        self.is_loaded = True

    def _get_model_prediction(self, model, tfidf_vec) -> Dict[str, Any]:
        """
        Execute prediction on a model and extract mathematically valid confidence.
        Label mapping: 0 -> FAKE, 1 -> REAL.
        """
        pred_label_idx = int(model.predict(tfidf_vec)[0])
        label_str = "REAL" if pred_label_idx == 1 else "FAKE"

        confidence = 0.0
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(tfidf_vec)[0]
            confidence = float(probs[pred_label_idx])
        else:
            confidence = 1.0

        return {
            "prediction": label_str,
            "label_idx": pred_label_idx,
            "confidence": round(confidence * 100, 2)
        }

    def predict(self, raw_text: str, save_to_db: bool = True) -> Dict[str, Any]:
        """
        Analyze news text through NLP preprocessing, TF-IDF feature extraction,
        evaluates Logistic Regression and Multinomial Naive Bayes, and applies consensus rules.

        Rules:
          - Both predict REAL -> REAL
          - Both predict FAKE -> FAKE
          - Disagree -> UNCERTAIN
        """
        start_time = time.perf_counter()

        if not raw_text or not raw_text.strip():
            return {
                "error": "Empty input text. Please provide news text to analyze.",
                "status": "error"
            }

        # Step 1: Preprocessing
        cleaned = clean_text(raw_text)
        if not cleaned:
            return {
                "error": "Text contains only stopwords or invalid characters after cleaning.",
                "status": "error"
            }

        # Step 2: TF-IDF Transformation
        tfidf_vec = self.vectorizer.transform([cleaned])

        # Step 3: Model Inferences (2 Models)
        lr_res = self._get_model_prediction(self.model_lr, tfidf_vec)
        nb_res = self._get_model_prediction(self.model_nb, tfidf_vec)

        # Step 4: Compare Predictions
        lr_pred = lr_res["prediction"]
        nb_pred = nb_res["prediction"]

        if lr_pred == "REAL" and nb_pred == "REAL":
            final_prediction = "REAL"
            vote_tally = "Both models agree: REAL"
        elif lr_pred == "FAKE" and nb_pred == "FAKE":
            final_prediction = "FAKE"
            vote_tally = "Both models agree: FAKE"
        else:
            final_prediction = "UNCERTAIN"
            vote_tally = f"Models disagree ({lr_pred} vs {nb_pred})"

        duration_ms = (time.perf_counter() - start_time) * 1000.0

        # Step 5: Save to SQLite database
        db_id = None
        if save_to_db:
            try:
                db_id = insert_prediction(
                    news_text=raw_text.strip(),
                    clean_text=cleaned,
                    lr_pred=lr_pred,
                    nb_pred=nb_pred,
                    final_pred=final_prediction,
                    lr_conf=lr_res["confidence"],
                    nb_conf=nb_res["confidence"],
                    analysis_duration_ms=duration_ms
                )
            except Exception as e:
                print(f"[Warning] Failed to log prediction to SQLite: {e}")

        return {
            "status": "success",
            "id": db_id,
            "final_prediction": final_prediction,
            "vote_tally": vote_tally,
            "models": {
                "logistic_regression": lr_res,
                "naive_bayes": nb_res
            },
            "cleaned_text_preview": cleaned[:200] + ("..." if len(cleaned) > 200 else ""),
            "duration_ms": round(duration_ms, 2),
            "disclaimer": "This prediction is generated using pattern recognition from trained machine learning models (Logistic Regression and Multinomial Naive Bayes) and does not represent official journalistic verification."
        }


# Singleton instance
_predictor_instance = None


def get_predictor() -> FakeNewsPredictor:
    """Get or create singleton FakeNewsPredictor instance."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = FakeNewsPredictor()
    return _predictor_instance


if __name__ == "__main__":
    predictor = FakeNewsPredictor()
    sample = "The Federal Reserve announced steady interest rates after policy meeting."
    print("Test prediction:", predictor.predict(sample, save_to_db=False))
