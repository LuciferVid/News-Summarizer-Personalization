"""Embedding pipeline using sentence-transformers."""

from __future__ import annotations

from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session

from database.models import Article


class EmbeddingService:
    """Singleton-like wrapper around sentence-transformers model."""

    _model: SentenceTransformer | None = None

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        if EmbeddingService._model is None:
            EmbeddingService._model = SentenceTransformer(model_name)
        self.model = EmbeddingService._model
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()

    def embed_text(self, text: str) -> list[float]:
        """Returns vector embedding for the given text."""
        return self.model.encode(text, normalize_embeddings=True).tolist()


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

