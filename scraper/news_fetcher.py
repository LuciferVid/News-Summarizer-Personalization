"""Fetches real-time news from GNews and stores it in SQLite."""

from __future__ import annotations

import os
import re
from datetime import datetime

from dotenv import load_dotenv
import requests
from sqlalchemy.orm import Session

from database.models import Article, SessionLocal

load_dotenv()

CATEGORIES = ["technology", "sports", "business", "health", "world", "nation"]
COUNTRIES = ["in", "us", "gb"]
GNEWS_TOP_HEADLINES_URL = "https://gnews.io/api/v4/top-headlines"


def clean_content(text: str) -> str:
    """Removes basic HTML tags and normalizes whitespace."""
    no_html = re.sub(r"<[^>]+>", " ", text or "")
    no_urls = re.sub(r"http\S+", " ", no_html)
    return re.sub(r"\s+", " ", no_urls).strip()


def _parse_published_at(value: str | None) -> datetime:
    if not value:
        return datetime.utcnow()
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return datetime.utcnow()


def fetch_and_store_news() -> int:
    """Fetches top headlines by country/category and stores new articles only."""
    api_key = os.getenv("GNEWS_API_KEY") or os.getenv("NEWS_API_KEY")
    if not api_key:
        print("GNEWS_API_KEY is missing. Skipping news fetch.")
        return 0

    db: Session = SessionLocal()
    new_count = 0
    try:
        for country in COUNTRIES:
            for category in CATEGORIES:
                params = {
                    "token": api_key,
                    "lang": "en",
                    "country": country,
                    "topic": category,
                    "max": 10,
                }
                try:
                    response = requests.get(GNEWS_TOP_HEADLINES_URL, params=params, timeout=30)
                    response.raise_for_status()
                    data = response.json()
                except Exception as exc:
                    print(f"Failed fetch for country={country} category={category}: {exc}")
                    continue

                for item in data.get("articles", []):
                    url = item.get("url")
                    if not url:
                        continue

                    exists = db.query(Article).filter(Article.url == url).first()
                    if exists:
                        continue

                    title = (item.get("title") or "Untitled").strip()
                    raw_content = item.get("content") or item.get("description") or ""
                    content = clean_content(raw_content)
                    if not content:
                        content = clean_content(f"{title}. {item.get('description') or ''}")

                    article = Article(
                        title=title[:500],
                        content=content,
                        source=(item.get("source") or {}).get("name", "Unknown"),
                        category=category,
                        url=url,
                        published_at=_parse_published_at(item.get("publishedAt")),
                    )
                    db.add(article)
                    new_count += 1

        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()

    return new_count

