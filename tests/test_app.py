"""
End-to-End Test Suite for AI-Based Fake News Detection System.

Tests NLP preprocessing, models, database operations, and Flask web routes.
"""

import os
import sys
import unittest
import json

# Ensure project root is in system path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app import app
from src.preprocessing import clean_text
from src.prediction import get_predictor
from src.database import (
    init_db,
    insert_prediction,
    get_history,
    get_dashboard_stats,
    clear_history
)


class TestFakeNewsDetectionSystem(unittest.TestCase):
    """Complete test suite for fake news detection pipeline and Flask app."""

    @classmethod
    def setUpClass(cls):
        init_db()
        cls.predictor = get_predictor()
        cls.client = app.test_client()

    def test_01_nlp_preprocessing_negation_preservation(self):
        """Verify negation words are preserved while stopwords are removed."""
        raw = "The senator will not support the proposed bill and never voted for it."
        cleaned = clean_text(raw)
        self.assertIn("not", cleaned, "Negation 'not' must be preserved")
        self.assertIn("never", cleaned, "Negation 'never' must be preserved")
        self.assertNotIn("will", cleaned, "Common stopword 'will' should be removed")
        self.assertNotIn("the", cleaned, "Common stopword 'the' should be removed")

    def test_02_nlp_preprocessing_leakage_and_urls(self):
        """Verify wire service prefixes and URLs are stripped."""
        raw = "WASHINGTON (Reuters) - Scientists report new breakthrough at https://example.com/test"
        cleaned = clean_text(raw)
        self.assertNotIn("reuters", cleaned)
        self.assertNotIn("https", cleaned)
        self.assertIn("scientists", cleaned)

    def test_03_prediction_real_news(self):
        """Verify prediction on a genuine news sample with 2 models."""
        real_text = (
            "The Federal Reserve announced on Wednesday that it will hold its benchmark interest rate "
            "steady following a two-day policy meeting, citing sustained moderate economic growth "
            "and low unemployment across domestic labor markets while continuing to monitor global supply chain developments."
        )
        res = self.predictor.predict(real_text, save_to_db=False)
        self.assertEqual(res["status"], "success")
        self.assertIn(res["final_prediction"], ["REAL", "FAKE", "UNCERTAIN"])
        self.assertIn("logistic_regression", res["models"])
        self.assertIn("naive_bayes", res["models"])
        self.assertNotIn("svm", res["models"])
        self.assertGreater(res["duration_ms"], 0)

    def test_04_prediction_fake_news(self):
        """Verify prediction on a sensational fake news sample."""
        fake_text = (
            "BREAKING BOMBSHELL: Whistleblowers have leaked classified footage exposing a secret underground bunker "
            "beneath Washington where deep state operatives have been genetically modifying reptilian aliens "
            "to manipulate presidential election ballots and control international weather patterns!"
        )
        res = self.predictor.predict(fake_text, save_to_db=False)
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["final_prediction"], "FAKE")
        self.assertEqual(res["models"]["logistic_regression"]["prediction"], "FAKE")
        self.assertEqual(res["models"]["naive_bayes"]["prediction"], "FAKE")

    def test_05_database_operations(self):
        """Verify SQLite insertion, stats query, and clearing."""
        # Insert test record
        record_id = insert_prediction(
            news_text="Unit test news sample text for SQLite verification.",
            clean_text="unit test news sample text sqlite verification",
            lr_pred="REAL",
            nb_pred="REAL",
            final_pred="REAL",
            lr_conf=95.0,
            nb_conf=90.0,
            analysis_duration_ms=12.5
        )
        self.assertIsNotNone(record_id)
        self.assertGreater(record_id, 0)

        # Retrieve history
        history = get_history(search_query="Unit test news sample", limit=5)
        self.assertGreater(len(history), 0)
        self.assertEqual(history[0]["final_prediction"], "REAL")

        # Check stats
        stats = get_dashboard_stats()
        self.assertGreaterEqual(stats["total_analyzed"], 1)
        self.assertGreaterEqual(stats["total_real"], 1)

    def test_06_flask_routes(self):
        """Verify all Flask HTTP routes return 200 OK."""
        # Home page
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"AI-Powered Fake News Detection", response.data)

        # Dashboard page
        response = self.client.get('/dashboard')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"System Dashboard", response.data)

        # History page
        response = self.client.get('/history')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Prediction History", response.data)

        # API Stats
        response = self.client.get('/api/stats')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("total_analyzed", data)

        # API Samples
        response = self.client.get('/api/samples')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)

    def test_07_predict_api_endpoint(self):
        """Verify /predict API endpoint via POST."""
        payload = {
            "news_text": "Scientists from the international observatory announced the discovery of an Earth-sized exoplanet orbiting within its star's habitable zone."
        }
        response = self.client.post(
            '/predict',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "success")
        self.assertIn("final_prediction", data)
        self.assertIn("vote_tally", data)
        self.assertIn("models", data)


if __name__ == '__main__':
    unittest.main()
