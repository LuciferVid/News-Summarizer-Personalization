"""Background jobs for periodic fetching and summarization."""

from __future__ import annotations

from apscheduler.schedulers.background import BackgroundScheduler

from database.models import SessionLocal
from pipeline.summarizer import run_summarization_pipeline
from scraper.news_fetcher import fetch_and_store_news

scheduler = BackgroundScheduler()


def _run_summarization_job() -> None:
    db = SessionLocal()
    try:
        run_summarization_pipeline(db)
    finally:
        db.close()


def start_scheduler() -> None:
    """Starts periodic jobs if scheduler is not already running."""
    if scheduler.running:
        return

    scheduler.add_job(
        fetch_and_store_news,
        "interval",
        minutes=15,
        id="fetch_news",
        replace_existing=True,
        misfire_grace_time=300,
    )
    scheduler.add_job(
        _run_summarization_job,
        "interval",
        minutes=18,
        id="summarize_news",
        replace_existing=True,
        misfire_grace_time=300,
    )
    scheduler.start()

