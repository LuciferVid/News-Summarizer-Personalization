"""RAG question answering over ingested news context."""

from __future__ import annotations
import os
import re
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from database.models import Article, SessionLocal
from database.vector_store import vector_store

load_dotenv()

# Minimum cosine similarity to consider an article relevant.
MIN_SIMILARITY = 0.35

RAG_PROMPT = """
You are a helpful news assistant with access to today's news articles.

Rules:
1. Answer the user's question using the context below.
2. If the EXACT topic isn't covered, find the CLOSEST related news from
   the context and share it. For example, if asked about weather in Pune
   and you have weather news from Delhi, share that and note the difference.
3. If the question is broad (e.g. "what's happening in sports?"), summarize
   the most interesting articles from the context.
4. ONLY say "I don't have news about this specific topic" if absolutely
   NONE of the articles are even loosely related — then suggest what
   topics ARE available based on the context.
5. Always cite the source article title(s) at the end.

Context:
{context}

Question: {question}

Answer (with source citations):
"""


def _keyword_fallback(db: Session, query: str, limit: int = 10) -> list[Article]:
    """Falls back to simple keyword matching when FAISS index is empty."""
    tokens = [t for t in re.findall(r"[a-zA-Z0-9]+", query.lower()) if len(t) >= 3]
    if not tokens:
        return []

    articles = (
        db.query(Article)
        .filter(Article.short_summary.isnot(None))
        .order_by(Article.published_at.desc())
        .limit(200)
        .all()
    )

    scored: list[tuple[float, Article]] = []
    for article in articles:
        text = f"{article.title} {article.short_summary or ''} {article.category}".lower()
        hits = sum(1 for t in tokens if t in text)
        if hits == 0:
            continue
        score = hits / len(tokens)
        scored.append((score, article))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [a for _, a in scored[:limit]]


def _latest_articles(db: Session, limit: int = 10) -> list[Article]:
    """Returns the most recent summarized articles as a last-resort context."""
    return (
        db.query(Article)
        .filter(Article.short_summary.isnot(None))
        .order_by(Article.published_at.desc())
        .limit(limit)
        .all()
    )


def answer_question(query: str, user_id: str) -> dict:
    """Answers a user query from relevant news context."""
    import google.generativeai as genai
    import json as _json
    _ = user_id  # Reserved for future user-aware retrieval.
    db: Session = SessionLocal()
    try:
        ordered_articles: list[Article] = []

        # 1. Try FAISS semantic search first.
        scored_results = vector_store.search_similar_with_scores(query, top_k=15)
        relevant_ids = [
            article_id
            for article_id, score in scored_results
            if score >= MIN_SIMILARITY
        ]

        if relevant_ids:
            articles = db.query(Article).filter(Article.id.in_(relevant_ids)).all()
            article_by_id = {a.id: a for a in articles}
            ordered_articles = [article_by_id[aid] for aid in relevant_ids if aid in article_by_id]

        # 2. If FAISS returned nothing, try keyword fallback.
        if not ordered_articles:
            print(f"[RAG] FAISS returned 0 relevant results (index={vector_store.index.ntotal}). Trying keyword fallback.")
            ordered_articles = _keyword_fallback(db, query)

        # 3. If keywords matched nothing either, use latest articles so the
        #    model can still offer something useful.
        if not ordered_articles:
            print("[RAG] Keyword fallback also empty. Using latest articles.")
            ordered_articles = _latest_articles(db)

        if not ordered_articles:
            return {
                "answer": "No news articles are available right now. Please try refreshing the news feed first.",
                "sources": [],
            }

        # Build context from pre-computed summaries — fast and quota-free.
        context_parts: list[str] = []
        sources: list[dict] = []
        seen_urls: set[str] = set()

        for article in ordered_articles:
            body_parts = []
            if article.short_summary:
                body_parts.append(article.short_summary)
            if article.bullets:
                try:
                    bullets = _json.loads(article.bullets)
                    body_parts.extend(f"- {b}" for b in bullets if b and b != "N/A")
                except Exception:
                    pass
            if not body_parts and article.content:
                body_parts.append(article.content[:800])

            if not body_parts:
                continue

            context_parts.append(f"Title: {article.title}\nCategory: {article.category}\n" + "\n".join(body_parts))
            if article.url not in seen_urls:
                sources.append({"title": article.title, "url": article.url})
                seen_urls.add(article.url)

        if not context_parts:
            return {
                "answer": "No news articles are available right now. Please try refreshing the news feed first.",
                "sources": [],
            }

        context = "\n\n".join(context_parts)

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return {
                "answer": "AI model is not configured. Please set GEMINI_API_KEY.",
                "sources": sources,
            }

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        prompt = RAG_PROMPT.format(context=context, question=query)
        try:
            response = model.generate_content(
                prompt,
                generation_config={"temperature": 0.3}
            )
            answer = (response.text or "").strip()
        except Exception as e:
            print(f"[RAG] Model generation failed: {e}")
            answer = "Sorry, the AI model is temporarily unavailable. Please try again in a moment."

        return {
            "answer": answer,
            "sources": sources,
        }
    finally:
        db.close()
