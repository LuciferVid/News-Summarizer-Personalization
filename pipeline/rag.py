"""RAG question answering over ingested news context."""

from __future__ import annotations
import os
from collections import defaultdict
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

def _chunk_article(article: Article) -> list[str]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    return splitter.split_text(article.content or "")

def answer_question(query: str, user_id: str) -> dict:
    """Answers a user query from relevant news context."""
    import google.generativeai as genai
    _ = user_id  # Reserved for future user-aware retrieval.
    db: Session = SessionLocal()
    try:
        # Increase top_k to 10 to ensure we don't miss the most relevant article 
        # due to initial vector search noise.
        article_ids = vector_store.search_similar(query, top_k=10)
        if not article_ids:
            return {
                "answer": "I don't have enough information in the current news to answer this.",
                "sources": [],
            }

        articles = db.query(Article).filter(Article.id.in_(article_ids)).all()
        article_by_id = {article.id: article for article in articles}
        ordered_articles = [article_by_id[aid] for aid in article_ids if aid in article_by_id]

        chunk_pool: list[tuple[float, str, Article]] = []
        embedding_service = vector_store.embedding_service
        
        raw_query_vec = embedding_service.embed_text(query)
        # Normalize query vector for cosine similarity
        q_norm = sum(x*x for x in raw_query_vec)**0.5
        query_embedding = [x/q_norm for x in raw_query_vec] if q_norm > 0 else raw_query_vec

        for article in ordered_articles:
            for chunk in _chunk_article(article):
                raw_chunk_vec = embedding_service.embed_text(chunk)
                # Normalize chunk vector
                c_norm = sum(x*x for x in raw_chunk_vec)**0.5
                chunk_vec = [x/c_norm for x in raw_chunk_vec] if c_norm > 0 else raw_chunk_vec
                
                # Cosine similarity
                score = sum(q * c for q, c in zip(query_embedding, chunk_vec))
                chunk_pool.append((score, chunk, article))

        chunk_pool.sort(key=lambda item: item[0], reverse=True)
        # Take slightly more chunks for better context
        top_chunks = chunk_pool[:8]

        grouped_context = defaultdict(list)
        for _, chunk, article in top_chunks:
            grouped_context[article.title].append(chunk)

        context_parts = []
        sources = []
        seen_urls = set()
        for title, chunks in grouped_context.items():
            context_parts.append(f"Title: {title}\n" + "\n".join(chunks))
            article = next((a for a in ordered_articles if a.title == title), None)
            if article and article.url not in seen_urls:
                sources.append({"title": article.title, "url": article.url})
                seen_urls.add(article.url)
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
            # Use lower temperature for more factual Retrieval-Augmented Generation
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
