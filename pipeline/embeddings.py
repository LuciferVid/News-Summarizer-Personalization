"""Embedding pipeline using sentence-transformers."""

from __future__ import annotations
from sqlalchemy.orm import Session
from database.models import Article

class EmbeddingService:
    """Uses Google Gemini for text embeddings."""

    def __init__(self) -> None:
        self.embedding_dimension = 768  # Dimension for text-embedding-004

    def embed_text(self, text: str) -> list[float]:
        """Returns vector embedding for the given text."""
        import os
        import google.generativeai as genai
        
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return [0.0] * self.embedding_dimension
            
        genai.configure(api_key=api_key)
        
        try:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
            )
            return result["embedding"]
        except Exception as e:
            print(f"[EmbeddingService] Failed: {e}")
            return [0.0] * self.embedding_dimension

def index_article(article: Article) -> None:
    """Embeds a summarized article and adds it to FAISS."""
    if not article.short_summary:
        return
    from database.vector_store import vector_store

    text = f"{article.title}\n{article.short_summary}"
    vector_store.add_article_to_index(article.id, text)

def build_missing_embeddings(db: Session) -> int:
    """Indexes all summarized articles that are not in FAISS yet."""
    from database.vector_store import vector_store

    articles = db.query(Article).filter(Article.short_summary.isnot(None)).all()
    count = 0
    for article in articles:
        if article.id not in vector_store.mapping:
            index_article(article)
            count += 1
    return count
