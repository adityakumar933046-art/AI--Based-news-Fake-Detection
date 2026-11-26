"""
Flask Backend Application for AI-Based Fake News Detection System.

Provides interactive web routes for:
  - Home Page with dynamic prediction analyzer and test samples
  - Prediction API with NLP preprocessing and Majority Voting
  - Dynamic Dashboard with live SQLite statistics and ML evaluation charts
  - Prediction History with search, filter, and clear history functionality
"""

import os
import sys
import json
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash

# Ensure project root is in system path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.database import (
    init_db,
    get_history,
    get_history_count,
    get_dashboard_stats,
    clear_history
)
from src.prediction import get_predictor
from src.evaluation import load_metrics

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "fake-news-detector-secret-key-2026")

# Auto-initialize SQLite database on launch
init_db()

# Preloaded sample news for instant testing
SAMPLE_NEWS = [
    {
        "id": "sample-real-1",
        "category": "REAL NEWS",
        "title": "Federal Reserve Holds Benchmark Rates Steady",
        "text": "The Federal Reserve announced on Wednesday that it will hold its benchmark interest rate steady following a two-day policy meeting, citing sustained moderate economic growth and low unemployment across domestic labor markets while continuing to monitor global supply chain developments."
    },
    {
        "id": "sample-real-2",
        "category": "REAL NEWS",
        "title": "NASA Rover Collects New Geological Core Samples",
        "text": "NASA's scientific exploration team confirmed today that the Perseverance rover has successfully drilled and stored its latest rock core sample from the ancient river delta on Mars, providing crucial data on historical planetary hydrology and geological formations."
    },
    {
        "id": "sample-fake-1",
        "category": "FAKE NEWS",
        "title": "Secret Underground Facility Breeding Alien Hybrids",
        "text": "BREAKING BOMBSHELL: Whistleblowers have leaked classified footage exposing a secret underground bunker beneath Washington where deep state operatives have been genetically modifying reptilian aliens to manipulate presidential election ballots and control international weather patterns!"
    },
    {
        "id": "sample-fake-2",
        "category": "FAKE NEWS",
        "title": "Drinking Distilled Lemon Battery Acid Cures All Illness",
        "text": "SHOCKING DISCOVERY: Big pharma executives were caught hiding the miraculous cure for all modern human ailments! Renowned rogue scientist reveals that drinking a teaspoon of distilled lemon juice mixed with battery electrolyte completely eliminates every known virus and pathogen in exactly 15 minutes!"
    }
]


@app.route('/')
def home():
    """Home page route displaying news analyzer, sample news, and prediction section."""
    return render_template('index.html', samples=SAMPLE_NEWS)


@app.route('/about')
def about():
    """About page route describing the AI system architecture and technologies."""
    return render_template('about.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict endpoint for news text.
    Handles both JSON API requests and standard form submissions.
    """
    raw_text = ""
    if request.is_json:
        data = request.get_json() or {}
        raw_text = data.get('news_text', '')
    else:
        raw_text = request.form.get('news_text', '')

    raw_text = raw_text.strip()
    if not raw_text:
        error_msg = "Please enter news text to analyze."
        if request.is_json:
            return jsonify({"status": "error", "error": error_msg}), 400
        flash(error_msg, "warning")
        return redirect(url_for('home'))

    try:
        predictor = get_predictor()
        result = predictor.predict(raw_text, save_to_db=True)
        
        if result.get("status") == "error":
            if request.is_json:
                return jsonify(result), 400
            flash(result.get("error", "Error processing text"), "danger")
            return redirect(url_for('home'))

        if request.is_json:
            return jsonify(result)
        
        return render_template('index.html', samples=SAMPLE_NEWS, prediction_result=result, input_text=raw_text)

    except Exception as e:
        error_msg = f"Prediction service error: {str(e)}"
        if request.is_json:
            return jsonify({"status": "error", "error": error_msg}), 500
        flash(error_msg, "danger")
        return redirect(url_for('home'))


@app.route('/dashboard')
def dashboard():
    """
    Dynamic analytics dashboard showing live SQLite database metrics
    and real ML evaluation data from model training.
    """
    db_stats = get_dashboard_stats()
    metrics_path = os.path.join(PROJECT_ROOT, 'models', 'evaluation_metrics.json')
    evaluation_metrics = load_metrics(metrics_path)

    return render_template(
        'dashboard.html',
        stats=db_stats,
        metrics=evaluation_metrics
    )


@app.route('/history')
def history():
    """
    Prediction history page displaying stored SQLite predictions with
    search, category filtering, and pagination.
    """
    search_query = request.args.get('search', '').strip()
    filter_pred = request.args.get('filter', 'ALL').strip()
    page = max(1, int(request.args.get('page', 1)))
    per_page = 20
    offset = (page - 1) * per_page

    total_records = get_history_count(search_query=search_query, filter_pred=filter_pred)
    records = get_history(
        search_query=search_query,
        filter_pred=filter_pred,
        limit=per_page,
        offset=offset
    )

    total_pages = max(1, (total_records + per_page - 1) // per_page)

    return render_template(
        'history.html',
        records=records,
        total_records=total_records,
        current_page=page,
        total_pages=total_pages,
        search_query=search_query,
        current_filter=filter_pred
    )


@app.route('/clear-history', methods=['POST'])
def handle_clear_history():
    """Clear all records from prediction history in SQLite database."""
    try:
        deleted_count = clear_history()
        msg = f"Successfully cleared {deleted_count} prediction history record{'s' if deleted_count != 1 else ''}."
        if request.is_json:
            return jsonify({"status": "success", "message": msg, "deleted_count": deleted_count})
        flash(msg, "success")
    except Exception as e:
        msg = f"Failed to clear history: {str(e)}"
        if request.is_json:
            return jsonify({"status": "error", "message": msg}), 500
        flash(msg, "danger")

    return redirect(url_for('history'))


@app.route('/api/stats')
def api_stats():
    """API endpoint for live dashboard data updates."""
    stats = get_dashboard_stats()
    return jsonify(stats)


@app.route('/api/samples')
def api_samples():
    """API endpoint providing pre-configured test news samples."""
    return jsonify(SAMPLE_NEWS)


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("AI-Based Fake News Detection System is starting...")
    print("Flask server running at http://127.0.0.1:5000 (0.0.0.0:5000)")
    print("=" * 60 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
