"""
Ensemble retrieval combining dense (FAISS) and sparse (TF-IDF) methods.
"""

import numpy as np
from typing import List, Optional


class EnsembleRetriever:
    """
    Ensemble retriever combining dense and sparse retrieval.

    Uses FAISS for dense retrieval with sentence embeddings and
    TF-IDF with character n-grams for sparse retrieval. The final
    candidates are the union of both methods.

    Parameters
    ----------
    embedding_model : str
        Sentence-transformer model name for dense embeddings.
    top_k : int
        Number of candidates to retrieve from each method.
    device : str, optional
        Device for embedding model ("cuda" or "cpu").
    cache_dir : str, optional
        Directory to download/cache models. Defaults to HuggingFace cache.
    """

    def __init__(
        self,
        embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
        top_k: int = 20,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        self.embedding_model_name = embedding_model
        self.top_k = top_k
        self.device = device
        self.cache_dir = cache_dir

        self._embed_model = None
        self._faiss_index = None
        self._tfidf_vectorizer = None
        self._tfidf_matrix = None
        self._corpus_texts: List[str] = []

    def _load_embedding_model(self):
        """Lazy load the embedding model."""
        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer

            self._embed_model = SentenceTransformer(
                self.embedding_model_name,
                device=self.device,
                cache_folder=self.cache_dir,
            )

    def index(self, texts: List[str], show_progress: bool = True) -> None:
        """
        Build retrieval indices for the corpus.

        Parameters
        ----------
        texts : List[str]
            List of corpus texts to index.
        show_progress : bool
            Whether to show progress bar for embedding generation.
        """
        import faiss
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._corpus_texts = texts

        # Build dense index
        self._load_embedding_model()

        embeddings = self._embed_model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        ).astype(np.float32)

        dim = embeddings.shape[1]
        self._faiss_index = faiss.IndexFlatIP(dim)
        self._faiss_index.add(embeddings)

        # Build sparse index
        self._tfidf_vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 4),
            lowercase=True,
            max_features=50000,
        )
        self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(texts)

    def retrieve(self, query: str) -> List[int]:
        """
        Retrieve candidate indices for a query.

        Parameters
        ----------
        query : str
            The query text.

        Returns
        -------
        List[int]
            Indices of candidate matches in the corpus.
        """
        if self._faiss_index is None:
            raise ValueError("Must call index() before retrieve()")

        # Dense retrieval
        query_emb = self._embed_model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        _, dense_indices = self._faiss_index.search(query_emb, self.top_k)
        dense_set = set(dense_indices[0].tolist())

        # Sparse retrieval
        from sklearn.metrics.pairwise import cosine_similarity

        query_vec = self._tfidf_vectorizer.transform([query])
        sparse_scores = cosine_similarity(query_vec, self._tfidf_matrix)[0]
        sparse_indices = np.argsort(-sparse_scores)[: self.top_k]
        sparse_set = set(sparse_indices.tolist())

        # Ensemble: union of both
        combined = list(dense_set | sparse_set)

        # Remove invalid indices (e.g., -1 from FAISS if corpus is small)
        combined = [i for i in combined if 0 <= i < len(self._corpus_texts)]

        return combined
