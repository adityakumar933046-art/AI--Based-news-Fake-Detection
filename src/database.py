"""
Database Management Module using SQLite.

Handles initialization, schema migration, inserting prediction records,
query filtering, search, pagination, clear history, and aggregated statistics for the dashboard.
Supports 2-model architecture: Logistic Regression & Multinomial Naive Bayes.
"""

import os
import sqlite3
from typing import List, Dict, Any, Optional

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'database', 'database.db')


def get_db_connection():
    """Create and return a thread-safe connection to the SQLite database with dictionary rows."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """
    Initialize SQLite database and ensure tables and indexes exist.
    Handles migration if table was previously created with the 3-model schema.
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()

        # Check existing table columns for migration
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'")
        table_exists = cursor.fetchone() is not None

        if table_exists:
            cursor.execute("PRAGMA table_info(predictions)")
            cols = [col[1] for col in cursor.fetchall()]
            if 'svm_prediction' in cols:
                # Migrate from 3-model schema to 2-model schema
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS predictions_new (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        news_text TEXT NOT NULL,
                        clean_text TEXT,
                        lr_prediction TEXT NOT NULL,
                        nb_prediction TEXT NOT NULL,
                        final_prediction TEXT NOT NULL,
                        lr_confidence REAL DEFAULT 0.0,
                        nb_confidence REAL DEFAULT 0.0,
                        analysis_duration_ms REAL DEFAULT 0.0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                cursor.execute("""
                    INSERT INTO predictions_new (
                        id, news_text, clean_text,
                        lr_prediction, nb_prediction,
                        final_prediction,
                        lr_confidence, nb_confidence,
                        analysis_duration_ms, created_at
                    )
                    SELECT id, news_text, clean_text,
                           lr_prediction, nb_prediction,
                           final_prediction,
                           lr_confidence, nb_confidence,
                           analysis_duration_ms, created_at
                    FROM predictions
                """)
                cursor.execute("DROP TABLE predictions")
                cursor.execute("ALTER TABLE predictions_new RENAME TO predictions")
        else:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    news_text TEXT NOT NULL,
                    clean_text TEXT,
                    lr_prediction TEXT NOT NULL,
                    nb_prediction TEXT NOT NULL,
                    final_prediction TEXT NOT NULL,
                    lr_confidence REAL DEFAULT 0.0,
                    nb_confidence REAL DEFAULT 0.0,
                    analysis_duration_ms REAL DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_final ON predictions(final_prediction)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_created ON predictions(created_at)")
        conn.commit()
    finally:
        conn.close()


def insert_prediction(
    news_text: str,
    clean_text: str,
    lr_pred: str,
    nb_pred: str,
    final_pred: str,
    lr_conf: float = 0.0,
    nb_conf: float = 0.0,
    analysis_duration_ms: float = 0.0
) -> int:
    """
    Insert a newly evaluated prediction record into the database.
    Returns the new record ID.
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions (
                news_text, clean_text,
                lr_prediction, nb_prediction,
                final_prediction,
                lr_confidence, nb_confidence,
                analysis_duration_ms, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now', 'localtime'))
        """, (
            news_text, clean_text,
            lr_pred, nb_pred,
            final_pred,
            round(lr_conf, 4), round(nb_conf, 4),
            round(analysis_duration_ms, 2)
        ))
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_history(
    search_query: Optional[str] = None,
    filter_pred: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """
    Retrieve historical predictions with optional search and category filters.
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        query = "SELECT * FROM predictions WHERE 1=1"
        params = []

        if filter_pred and filter_pred.upper() != 'ALL':
            query += " AND UPPER(final_prediction) = ?"
            params.append(filter_pred.upper())

        if search_query and search_query.strip():
            query += " AND (news_text LIKE ? OR final_prediction LIKE ?)"
            term = f"%{search_query.strip()}%"
            params.extend([term, term])

        query += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_history_count(
    search_query: Optional[str] = None,
    filter_pred: Optional[str] = None
) -> int:
    """Get count of matching history records for pagination."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        query = "SELECT COUNT(*) FROM predictions WHERE 1=1"
        params = []

        if filter_pred and filter_pred.upper() != 'ALL':
            query += " AND UPPER(final_prediction) = ?"
            params.append(filter_pred.upper())

        if search_query and search_query.strip():
            query += " AND (news_text LIKE ? OR final_prediction LIKE ?)"
            term = f"%{search_query.strip()}%"
            params.extend([term, term])

        cursor.execute(query, params)
        return cursor.fetchone()[0]
    finally:
        conn.close()


def get_dashboard_stats() -> Dict[str, Any]:
    """
    Compute real aggregated statistics directly from SQLite for dashboard.
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()

        # Total analyzed
        cursor.execute("SELECT COUNT(*) FROM predictions")
        total = cursor.fetchone()[0]

        # Total by prediction outcome
        cursor.execute("""
            SELECT UPPER(final_prediction), COUNT(*)
            FROM predictions
            GROUP BY UPPER(final_prediction)
        """)
        counts = dict(cursor.fetchall())
        fake_count = counts.get('FAKE', 0) or counts.get('FAKE NEWS', 0)
        real_count = counts.get('REAL', 0) or counts.get('REAL NEWS', 0)
        uncertain_count = counts.get('UNCERTAIN', 0)

        # Average analysis duration
        cursor.execute("SELECT AVG(analysis_duration_ms) FROM predictions")
        avg_duration = cursor.fetchone()[0] or 0.0

        # Timeline count (last 7 recorded dates)
        cursor.execute("""
            SELECT SUBSTR(created_at, 1, 10) as day, COUNT(*) as cnt
            FROM predictions
            GROUP BY day
            ORDER BY day DESC
            LIMIT 7
        """)
        timeline_rows = cursor.fetchall()
        timeline = [{"date": r[0], "count": r[1]} for r in reversed(timeline_rows)]

        # Recent 5 predictions
        cursor.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 5")
        recent = [dict(r) for r in cursor.fetchall()]

        return {
            "total_analyzed": total,
            "total_fake": fake_count,
            "total_real": real_count,
            "total_uncertain": uncertain_count,
            "avg_duration_ms": round(avg_duration, 1),
            "timeline": timeline,
            "recent_predictions": recent
        }
    finally:
        conn.close()


def clear_history() -> int:
    """
    Clear all records from the predictions table.
    Returns number of deleted rows.
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM predictions")
        count = cursor.fetchone()[0]
        cursor.execute("DELETE FROM predictions")
        cursor.execute("DELETE FROM sqlite_sequence WHERE name='predictions'")
        conn.commit()
        return count
    finally:
        conn.close()


if __name__ == "__main__":
    init_db()
    print("Database initialized/migrated successfully at:", DB_PATH)
    print("Dashboard stats:", get_dashboard_stats())
