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
You are a helpful news assistant. Answer the user's question using the
context provided below.  Draw on ALL articles that are even partially
relevant — for example, if the user asks about weather in one city and
you have weather news from a nearby region, mention it and note the
difference.

If NONE of the articles are even remotely related to the question,
say: "I don't have enough information in the current news to answer this."

Always cite the source article title(s) you used at the end.

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


def answer_question(query: str, user_id: str) -> dict:
    """Answers a user query from relevant news context."""
    import google.generativeai as genai
    import json as _json
    _ = user_id  # Reserved for future user-aware retrieval.
    db: Session = SessionLocal()
    try:
        # Try FAISS first.
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
        else:
            # FAISS empty or no relevant results — fall back to keyword search.
            print(f"[RAG] FAISS returned 0 relevant results (index size={vector_store.index.ntotal}). Falling back to keyword search.")
            ordered_articles = _keyword_fallback(db, query)

        if not ordered_articles:
            return {
                "answer": "I don't have enough information in the current news to answer this.",
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

            context_parts.append(f"Title: {article.title}\n" + "\n".join(body_parts))
            if article.url not in seen_urls:
                sources.append({"title": article.title, "url": article.url})
                seen_urls.add(article.url)

        if not context_parts:
            return {
                "answer": "I don't have enough information in the current news to answer this.",
                "sources": [],
            }

        context = "\n\n".join(context_parts)

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return {
                "answer": "I don't have enough information in the current news to answer this.",
                "sources": sources,
            }

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        prompt = RAG_PROMPT.format(context=context, question=query)
        try:
            response = model.generate_content(
                prompt,
                generation_config={"temperature": 0.2}
            )
            answer = (response.text or "").strip()
        except Exception as e:
            print(f"[RAG] Model generation failed: {e}")
            answer = "I don't have enough information in the current news to answer this."

        return {
            "answer": answer,
            "sources": sources,
        }
    finally:
        db.close()
