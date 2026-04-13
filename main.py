"""FastAPI backend for AI-powered news summarizer."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from sqlalchemy.orm import Session

from database.models import Article, SessionLocal, UserInteraction, get_db, init_db
from database.vector_store import vector_store
from personalization.recommender import get_personalized_feed, update_preference
from pipeline.embeddings import build_missing_embeddings
from pipeline.rag import answer_question
from pipeline.summarizer import run_summarization_pipeline
from scraper.news_fetcher import fetch_and_store_news
from scraper.scheduler import start_scheduler

load_dotenv()

app = FastAPI(title="AI-Powered News Summarizer & Personalization Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    query: str
    user_id: str


class InteractRequest(BaseModel):
    user_id: str
    article_id: int
    interaction_type: str


@app.on_event("startup")
def startup_event() -> None:
    """Initializes DB, loads vector index, and starts scheduler."""
    init_db()
    vector_store.load()
    db = SessionLocal()
    try:
        build_missing_embeddings(db)
    finally:
        db.close()
    start_scheduler()


@app.get("/news/feed")
def news_feed(user_id: str, db: Session = Depends(get_db)) -> list[dict]:
    """Returns personalized feed for the given user."""
    articles = get_personalized_feed(db, user_id)
    result = []
    for article in articles:
        result.append(
            {
                "id": article.id,
                "title": article.title,
                "one_liner": article.one_liner,
                "source": article.source,
                "category": article.category,
                "url": article.url,
                "published_at": article.published_at.isoformat(),
                "short_summary": article.short_summary,
                "bullets": json.loads(article.bullets) if article.bullets else [],
            }
        )
    return result


@app.post("/news/ask")
def ask_news(payload: AskRequest) -> dict:
    """Answers user questions with RAG over indexed articles."""
    return answer_question(payload.query, payload.user_id)


@app.post("/user/interact")
def user_interact(payload: InteractRequest, db: Session = Depends(get_db)) -> dict:
    """Records user interaction and updates preference weights."""
    article = db.query(Article).filter(Article.id == payload.article_id).first()
    if not article:
        raise HTTPException(status_code=404, detail="Article not found.")

    if payload.interaction_type not in {"read", "liked", "skipped"}:
        raise HTTPException(status_code=400, detail="Invalid interaction_type.")

    interaction = UserInteraction(
        user_id=payload.user_id,
        article_id=payload.article_id,
        interaction_type=payload.interaction_type,
    )
    db.add(interaction)
    db.commit()

    update_preference(db, payload.user_id, article.category, payload.interaction_type)
    return {"status": "ok"}


@app.get("/news/search")
def search_news(q: str, db: Session = Depends(get_db)) -> list[dict]:
    """Performs hybrid semantic + keyword search for higher precision."""
    query = q.strip()
    if not query:
        return []

    semantic_results = vector_store.search_similar_with_scores(query, top_k=50)
    semantic_map = {article_id: score for article_id, score in semantic_results}
    query_tokens = [token for token in re.findall(r"[a-zA-Z0-9]+", query.lower()) if len(token) >= 3]

    recent_articles = db.query(Article).order_by(Article.published_at.desc()).limit(300).all()
    by_id = {article.id: article for article in recent_articles}

    candidate_ids = set(semantic_map.keys())
    for article in recent_articles:
        text = f"{article.title} {article.short_summary or ''} {article.source} {article.category}".lower()
        if query_tokens and any(token in text for token in query_tokens):
            candidate_ids.add(article.id)

    scored_candidates: list[tuple[float, Article]] = []
    for article_id in candidate_ids:
        article = by_id.get(article_id)
        if not article:
            continue

        semantic_score = semantic_map.get(article_id, 0.0)
        text = f"{article.title} {article.short_summary or ''} {article.content or ''} {article.source}".lower()
        keyword_hits = sum(1 for token in query_tokens if token in text)
        lexical_score = (keyword_hits / len(query_tokens)) if query_tokens else 0.0
        combined_score = (0.65 * semantic_score) + (0.35 * lexical_score)

        # Guardrail for named-entity style queries: require some lexical support.
        if len(query_tokens) >= 2 and lexical_score == 0.0 and semantic_score < 0.45:
            continue
        if combined_score < 0.20:
            continue
        scored_candidates.append((combined_score, article))

    scored_candidates.sort(key=lambda item: item[0], reverse=True)
    ordered = [article for _, article in scored_candidates[:10]]

    if not ordered:
        # Fallback: if local semantic index has no relevant documents,
        # fetch live keyword matches from GNews so users still get results.
        gnews_key = os.getenv("GNEWS_API_KEY") or os.getenv("NEWS_API_KEY")
        if not gnews_key:
            return []
        try:
            response = requests.get(
                "https://gnews.io/api/v4/search",
                params={"q": query, "lang": "en", "max": 10, "token": gnews_key},
                timeout=20,
            )
            response.raise_for_status()
            items = response.json().get("articles", [])
        except Exception:
            return []

        fallback_results = []
        for item in items:
            title = item.get("title") or "Untitled"
            source_name = (item.get("source") or {}).get("name", "Unknown")
            published_at = item.get("publishedAt")
            if published_at:
                try:
                    published_at = datetime.fromisoformat(published_at.replace("Z", "+00:00")).isoformat()
                except ValueError:
                    published_at = datetime.utcnow().isoformat()
            else:
                published_at = datetime.utcnow().isoformat()
            fallback_results.append(
                {
                    "id": None,
                    "title": title,
                    "one_liner": item.get("description"),
                    "source": source_name,
                    "category": "world",
                    "url": item.get("url"),
                    "published_at": published_at,
                    "short_summary": item.get("content") or item.get("description"),
                    "bullets": [],
                }
            )
        return fallback_results

    return [
        {
            "id": article.id,
            "title": article.title,
            "one_liner": article.one_liner,
            "source": article.source,
            "category": article.category,
            "url": article.url,
            "published_at": article.published_at.isoformat(),
            "short_summary": article.short_summary,
            "bullets": json.loads(article.bullets) if article.bullets else [],
        }
        for article in ordered
    ]


@app.post("/news/refresh")
def refresh_news(db: Session = Depends(get_db)) -> dict:
    """Manually triggers fetch + summarize pipelines."""
    new_articles = fetch_and_store_news()
    summarized = run_summarization_pipeline(db)
    return {
        "new_articles": new_articles,
        "summarized_articles": summarized,
    }

