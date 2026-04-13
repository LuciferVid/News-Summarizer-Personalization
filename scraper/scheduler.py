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

    scheduler.add_job(fetch_and_store_news, "interval", minutes=30, id="fetch_news", replace_existing=True)
    scheduler.add_job(
        _run_summarization_job,
        "interval",
        minutes=35,
        id="summarize_news",
        replace_existing=True,
    )
    scheduler.start()

