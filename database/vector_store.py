"""FAISS-backed vector store for article embeddings."""

from __future__ import annotations

import json
import os
from typing import List

import faiss
import numpy as np

from pipeline.embeddings import EmbeddingService

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_PATH = os.path.join(BASE_DIR, "faiss_index.bin")
MAPPING_PATH = os.path.join(BASE_DIR, "faiss_mapping.json")


class VectorStore:
    """Manages FAISS index and article-id mapping.

    Uses IndexFlatIP (inner product) on L2-normalised vectors so that
    the dot-product equals cosine similarity and scores are in [0, 1].
    """

    def __init__(self) -> None:
        self.embedding_service = EmbeddingService()
        self.dim = self.embedding_service.embedding_dimension
        self.index = faiss.IndexFlatIP(self.dim)
        self.mapping: list[int] = []
        self.load()

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        """L2-normalises a 2-D row vector in-place and returns it."""
        faiss.normalize_L2(vector)
        return vector

    def load(self) -> None:
        """Loads index/mapping from disk if available."""
        if os.path.exists(INDEX_PATH) and os.path.exists(MAPPING_PATH):
            try:
                temp_index = faiss.read_index(INDEX_PATH)
                if temp_index.d != self.dim:
                    print(f"[vector_store] Dimension mismatch ({temp_index.d} != {self.dim}), rebuilding index!")
                    raise ValueError("Dimension mismatch")
                # Accept only IndexFlatIP; rebuild if old L2 index is on disk.
                if not isinstance(temp_index, faiss.IndexFlatIP):
                    print("[vector_store] Old L2 index detected, rebuilding as IP index for cosine similarity.")
                    raise ValueError("Wrong index type")
                self.index = temp_index
                with open(MAPPING_PATH, "r", encoding="utf-8") as file:
                    self.mapping = json.load(file)
            except Exception:
                self.index = faiss.IndexFlatIP(self.dim)
                self.mapping = []

    def save(self) -> None:
        """Persists index and mapping to disk."""
        faiss.write_index(self.index, INDEX_PATH)
        with open(MAPPING_PATH, "w", encoding="utf-8") as file:
            json.dump(self.mapping, file)

    def add_article_to_index(self, article_id: int, text: str) -> None:
        """Adds an article embedding to FAISS and saves index."""
        if article_id in self.mapping:
            return

        embedding = self.embedding_service.embed_text(text)
        vector = self._normalize(np.array([embedding], dtype="float32"))
        self.index.add(vector)
        self.mapping.append(article_id)
        self.save()

    def search_similar(self, query_text: str, top_k: int = 5) -> List[int]:
        """Searches for similar article IDs given query text."""
        results = self.search_similar_with_scores(query_text, top_k=top_k)
        return [article_id for article_id, _ in results]

    def search_similar_with_scores(self, query_text: str, top_k: int = 5) -> List[tuple[int, float]]:
        """Returns (article_id, cosine_similarity) pairs for the query.

        Scores are in [0, 1]; higher means more similar.
        """
        if self.index.ntotal == 0:
            return []

        query_embedding = self.embedding_service.embed_text(query_text)
        query_vector = self._normalize(np.array([query_embedding], dtype="float32"))

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vector, k)
        scored_ids: list[tuple[int, float]] = []
        for pos, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.mapping):
                # IP on normalised vectors == cosine similarity in [-1, 1].
                # Clamp to [0, 1] to discard anti-correlated results.
                similarity = max(0.0, float(scores[0][pos]))
                scored_ids.append((self.mapping[idx], similarity))
        return scored_ids


vector_store = VectorStore()
