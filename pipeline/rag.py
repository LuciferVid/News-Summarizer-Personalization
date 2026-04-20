"""RAG question answering over ingested news context."""

from __future__ import annotations
import os
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from database.models import Article, SessionLocal
from database.vector_store import vector_store

load_dotenv()

RAG_PROMPT = """
You are a helpful news assistant. Answer the user's question using ONLY 
the context provided below. 
If the answer is not in the context, say exactly: 
"I don't have enough information in the current news to answer this."
Always end your answer by citing the source article title.

Context:
{context}

Question: {question}

Answer (with source citation at the end):
"""


def answer_question(query: str, user_id: str) -> dict:
    """Answers a user query from relevant news context."""
    import google.generativeai as genai
    import json as _json
    _ = user_id  # Reserved for future user-aware retrieval.
    db: Session = SessionLocal()
    try:
        article_ids = vector_store.search_similar(query, top_k=10)
        if not article_ids:
            return {
                "answer": "I don't have enough information in the current news to answer this.",
                "sources": [],
            }

        articles = db.query(Article).filter(Article.id.in_(article_ids)).all()
        article_by_id = {article.id: article for article in articles}
        ordered_articles = [article_by_id[aid] for aid in article_ids if aid in article_by_id]

        # Build context from pre-computed summaries — fast and quota-free.
        context_parts: list[str] = []
        sources: list[dict] = []
        seen_urls: set[str] = set()

        for article in ordered_articles:
            # Build the richest context we have from stored fields
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
                generation_config={"temperature": 0.1}
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
