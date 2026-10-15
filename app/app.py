"""
CHANAKYA - Flask Application
Experiment 10 (Part 2): Web interface wrapping the NLP pipeline.

Provides:
  GET  /          - Home page with article input form
  POST /analyze   - Runs the pipeline, saves result, shows the 5 outputs
  GET  /history   - Lists previously analyzed articles
"""

import sys
import os
import sqlite3
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request, redirect, url_for
from nlp.pipeline import ChanakyaPipeline

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# By default Flask looks for a 'static' folder INSIDE the same directory
# as this file (i.e. app/static/). Our static/ folder lives at the
# project root instead (a sibling of app/), so we must point Flask to
# it explicitly -- otherwise style.css silently fails to load and the
# page renders with zero styling (default browser HTML).
app = Flask(
    __name__,
    static_folder=os.path.join(PROJECT_ROOT, "static"),
    static_url_path="/static"
)

DB_PATH = os.path.join(PROJECT_ROOT, "db", "chanakya.db")

# Initialize the pipeline ONCE when the server starts (not per-request --
# training the classifier and loading the KB every request would be slow)
print("Starting CHANAKYA server -- initializing pipeline (this may take a moment)...")
pipeline = ChanakyaPipeline()


def init_db():
    """Creates the analyses table if it doesn't already exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            article_text TEXT NOT NULL,
            event_summary TEXT,
            why_in_news TEXT,
            background TEXT,
            connected_topics TEXT,
            important_entities TEXT,
            dates_mentioned TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def save_analysis(article_text: str, outputs: dict) -> int:
    """Saves a completed analysis to the database, returns the new row id."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO analyses
            (article_text, event_summary, why_in_news, background,
             connected_topics, important_entities, dates_mentioned)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        article_text,
        outputs["event_summary"],
        outputs["why_in_news"],
        outputs["background"],
        json.dumps(outputs["connected_topics"]),
        json.dumps(outputs["important_entities"]),
        json.dumps(outputs["dates_mentioned"]),
    ))
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    return new_id


def get_analysis(analysis_id: int) -> dict:
    """Retrieves a single saved analysis by id, formatted for template rendering."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,))
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None

    return {
        "id": row["id"],
        "article_text": row["article_text"],
        "event_summary": row["event_summary"],
        "why_in_news": row["why_in_news"],
        "background": row["background"],
        "connected_topics": json.loads(row["connected_topics"]),
        "important_entities": json.loads(row["important_entities"]),
        "dates_mentioned": json.loads(row["dates_mentioned"]),
        "created_at": row["created_at"],
    }


def get_all_analyses() -> list:
    """Retrieves a summary list of all past analyses for the history page."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT id, article_text, created_at FROM analyses ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "id": row["id"],
            "preview": row["article_text"].strip()[:120] + "...",
            "created_at": row["created_at"],
        }
        for row in rows
    ]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    article_text = request.form.get("article_text", "").strip()

    if not article_text:
        return render_template("index.html", error="Please paste an article to analyze.")

    outputs = pipeline.process_article(article_text)
    analysis_id = save_analysis(article_text, outputs)

    return redirect(url_for("view_result", analysis_id=analysis_id))


@app.route("/result/<int:analysis_id>")
def view_result(analysis_id):
    analysis = get_analysis(analysis_id)
    if analysis is None:
        return "Analysis not found.", 404
    return render_template("results.html", result=analysis)


@app.route("/history")
def history():
    analyses = get_all_analyses()
    return render_template("history.html", analyses=analyses)


if __name__ == "__main__":
    init_db()
    app.run(debug=True, port=5001)