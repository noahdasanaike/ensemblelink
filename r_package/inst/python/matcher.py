"""
Python backend for ensemblelink R package.
Zero-shot record linkage with ensemble retrieval and cross-encoder reranking.
"""

import numpy as np
from tqdm import tqdm
import torch


class EnsembleMatcher:
    """
    Zero-shot record linkage using ensemble retrieval + cross-encoder reranking.

    Parameters
    ----------
    embedding_model : str
        Sentence-transformers model for dense embeddings.
    reranker_model : str
        Cross-encoder model for reranking candidates.
    top_k : int
        Number of candidates to retrieve from each method.
    device : str
        Device for inference: "cuda", "cpu", or "auto".
    """

    def __init__(
        self,
        embedding_model="Qwen/Qwen3-Embedding-0.6B",
        reranker_model="jinaai/jina-reranker-v2-base-multilingual",
        top_k=30,
        device="auto"
    ):
        self.embedding_model_name = embedding_model
        self.reranker_model_name = reranker_model
        self.top_k = top_k

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Lazy loading
        self._embedding_model = None
        self._reranker_model = None
        self._faiss_index = None
        self._tfidf_vectorizer = None
        self._tfidf_matrix = None
        self._corpus = None

    def _load_embedding_model(self):
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(
                self.embedding_model_name,
                trust_remote_code=True
            )

    def _load_reranker_model(self):
        if self._reranker_model is None:
            from transformers import AutoModelForSequenceClassification

            self._reranker_model = AutoModelForSequenceClassification.from_pretrained(
                self.reranker_model_name,
                torch_dtype="auto",
                trust_remote_code=True,
            )
            self._reranker_model.to(self.device)
            self._reranker_model.eval()

    def index(self, corpus, show_progress=True):
        """
        Build retrieval indices for the corpus.

        Parameters
        ----------
        corpus : list of str
            Reference strings to match against.
        show_progress : bool
            Show progress bar during indexing.
        """
        import faiss
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._corpus = list(corpus)

        # Load embedding model
        self._load_embedding_model()

        # Dense embeddings
        embeddings = self._embedding_model.encode(
            self._corpus,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        ).astype(np.float32)

        # FAISS index
        dim = embeddings.shape[1]
        self._faiss_index = faiss.IndexFlatIP(dim)
        self._faiss_index.add(embeddings)

        # TF-IDF index
        self._tfidf_vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 4),
            lowercase=True,
            max_features=50000
        )
        self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(self._corpus)

    def _retrieve(self, query):
        """Retrieve candidates using ensemble (dense + sparse)."""
        from sklearn.metrics.pairwise import cosine_similarity

        # Dense retrieval
        query_emb = self._embedding_model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype(np.float32)
        _, dense_indices = self._faiss_index.search(query_emb, self.top_k)
        dense_set = set(dense_indices[0].tolist())

        # Sparse retrieval
        query_vec = self._tfidf_vectorizer.transform([query])
        sparse_scores = cosine_similarity(query_vec, self._tfidf_matrix)[0]
        sparse_indices = np.argsort(-sparse_scores)[:self.top_k]
        sparse_set = set(sparse_indices.tolist())

        # Union
        combined = list(dense_set | sparse_set)
        combined = [i for i in combined if 0 <= i < len(self._corpus)]
        return combined

    def _rerank(self, query, candidate_indices):
        """Rerank candidates using cross-encoder."""
        if not candidate_indices:
            return None, 0.0

        self._load_reranker_model()

        candidates = [self._corpus[i] for i in candidate_indices]

        # Use Jina v2's compute_score method
        pairs = [[query, c] for c in candidates]
        scores = self._reranker_model.compute_score(pairs, max_length=1024)

        # Find best match
        best_idx = int(np.argmax(scores))
        return candidates[best_idx], float(scores[best_idx])

    def match(self, queries, return_scores=False, show_progress=True):
        """
        Match queries to corpus.

        Parameters
        ----------
        queries : list of str
            Query strings to match.
        return_scores : bool
            If True, return (matches, scores) tuple.
        show_progress : bool
            Show progress bar.

        Returns
        -------
        matches : list of str
            Best matching corpus string for each query.
        scores : list of float (if return_scores=True)
            Match scores.
        """
        if self._corpus is None:
            raise ValueError("Must call index() before match()")

        matches = []
        scores = []

        iterator = tqdm(queries, desc="Matching") if show_progress else queries

        for query in iterator:
            candidate_indices = self._retrieve(query)
            match, score = self._rerank(query, candidate_indices)

            if match is None:
                match = self._corpus[0] if self._corpus else ""
                score = 0.0

            matches.append(match)
            scores.append(score)

        if return_scores:
            return matches, scores
        return matches

    def match_one(self, query, return_score=False):
        """
        Match a single query.

        Parameters
        ----------
        query : str
            Query string to match.
        return_score : bool
            If True, return (match, score) tuple.

        Returns
        -------
        match : str
            Best matching corpus string.
        score : float (if return_score=True)
            Match score.
        """
        candidate_indices = self._retrieve(query)
        match, score = self._rerank(query, candidate_indices)

        if match is None:
            match = self._corpus[0] if self._corpus else ""
            score = 0.0

        if return_score:
            return match, score
        return match
