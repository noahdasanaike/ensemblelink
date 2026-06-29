"""
Ensemble candidate retrieval combining dense and sparse methods.

Dense retrieval uses a sentence-transformer embedding (harrier by default) and
sparse retrieval uses character-n-gram TF-IDF. The two are complementary: the
embedding captures semantic and transliteration variation while the n-grams
capture surface-form overlap (shared substrings, abbreviations, typos). The
candidate pool for a query is the union of the top matches from each method.

Dense scores are computed by a direct inner product against the (normalized)
corpus embedding matrix rather than an approximate index, because the fusion
stage downstream needs the full dense score for every pooled candidate anyway.
"""

import numpy as np
from typing import List, Optional


def _topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the ``k`` largest scores, in descending order."""
    n = scores.shape[0]
    k = min(k, n)
    if k <= 0:
        return np.array([], dtype=int)
    if k < n:
        part = np.argpartition(-scores, k - 1)[:k]
        return part[np.argsort(-scores[part])]
    return np.argsort(-scores)


class EnsembleRetriever:
    """
    Ensemble retriever combining dense embeddings and sparse TF-IDF.

    Parameters
    ----------
    embedding_model : str
        Sentence-transformer model name for dense embeddings.
    pool_size : int
        Number of candidates to retrieve from each method (their union forms the
        pool). ``top_k`` is accepted as an alias for backward compatibility.
    ngram_range : tuple
        Character n-gram range for the sparse TF-IDF index.
    max_features : int
        Maximum TF-IDF vocabulary size.
    device : str, optional
        Device for the embedding model ("cuda" or "cpu").
    cache_dir : str, optional
        Directory to download/cache models. Defaults to the HuggingFace cache.
    """

    def __init__(
        self,
        embedding_model: str = "microsoft/harrier-oss-v1-0.6b",
        pool_size: Optional[int] = None,
        top_k: Optional[int] = None,
        ngram_range: tuple = (2, 4),
        max_features: int = 50000,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        self.embedding_model_name = embedding_model
        # pool_size is the primary name; top_k kept as an alias.
        self.pool_size = pool_size if pool_size is not None else (top_k if top_k is not None else 50)
        self.top_k = self.pool_size
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.device = device
        self.cache_dir = cache_dir

        self._embed_model = None
        self._tfidf_vectorizer = None
        self._tfidf_matrix = None
        self._tfidf_matrix_T = None
        self.corpus_embeddings: Optional[np.ndarray] = None
        self._corpus_texts: List[str] = []

    def _load_embedding_model(self):
        """Lazy-load the embedding model."""
        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer

            self._embed_model = SentenceTransformer(
                self.embedding_model_name,
                device=self.device,
                cache_folder=self.cache_dir,
                trust_remote_code=True,
            )

    def encode(
        self,
        texts: List[str],
        show_progress: bool = True,
        batch_size: int = 50000,
    ) -> np.ndarray:
        """Encode texts into L2-normalized float32 embeddings.

        Encoding is chunked so that large corpora do not have to be embedded in
        a single forward pass.
        """
        self._load_embedding_model()
        if len(texts) <= batch_size:
            return self._embed_model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
            ).astype(np.float32)

        first = self._embed_model.encode(
            texts[:1], normalize_embeddings=True, convert_to_numpy=True
        )
        dim = first.shape[1]
        out = np.empty((len(texts), dim), dtype=np.float32)
        out[0] = first[0]
        for start in range(1, len(texts), batch_size):
            end = min(start + batch_size, len(texts))
            out[start:end] = self._embed_model.encode(
                texts[start:end],
                normalize_embeddings=True,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
            ).astype(np.float32)
        return out

    def index(
        self,
        texts: List[str],
        show_progress: bool = True,
        batch_size: int = 50000,
    ) -> None:
        """Build the dense and sparse indices for the corpus.

        Parameters
        ----------
        texts : List[str]
            Corpus texts to index.
        show_progress : bool
            Show a progress bar while embedding.
        batch_size : int
            Number of texts to embed at once. Lower values reduce peak memory on
            large corpora. Default 50,000 works well up to ~1M records on a
            16 GB GPU; reduce to 10,000-20,000 for CPU-only environments.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._corpus_texts = list(texts)
        self.corpus_embeddings = self.encode(
            self._corpus_texts, show_progress=show_progress, batch_size=batch_size
        )

        self._tfidf_vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=self.ngram_range,
            lowercase=True,
            max_features=self.max_features,
        )
        self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(self._corpus_texts)
        self._tfidf_matrix_T = self._tfidf_matrix.T.tocsr()

    def _dense_scores(self, query_emb_row: np.ndarray) -> np.ndarray:
        """Cosine similarity of one query against every corpus row.

        Embeddings are L2-normalized, so the inner product is the cosine.
        """
        return self.corpus_embeddings @ query_emb_row

    def _sparse_scores(self, query_sparse_row) -> np.ndarray:
        """TF-IDF cosine of one query against every corpus row."""
        return (query_sparse_row @ self._tfidf_matrix_T).toarray().ravel()

    def pool(self, query_texts: List[str], show_progress: bool = True):
        """Retrieve the candidate pool and pooled scores for many queries.

        Parameters
        ----------
        query_texts : List[str]
            Queries to retrieve candidates for.

        Returns
        -------
        pools : list of list of int
            Candidate corpus indices per query (union of dense and sparse top-k).
        dense_pool : list of np.ndarray
            Dense cosine of each pooled candidate, per query.
        sparse_pool : list of np.ndarray
            Sparse TF-IDF cosine of each pooled candidate, per query.
        query_emb : np.ndarray
            The query embeddings (reused downstream for the CSLS correction).
        """
        if self.corpus_embeddings is None:
            raise ValueError("Must call index() before pool().")

        query_emb = self.encode(query_texts, show_progress=show_progress)
        query_sparse = self._tfidf_vectorizer.transform(query_texts)

        pools, dense_pool, sparse_pool = [], [], []
        iterator = range(len(query_texts))
        if show_progress:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="Pooling candidates")

        for qi in iterator:
            d = self._dense_scores(query_emb[qi])
            s = self._sparse_scores(query_sparse[qi])
            dense_top = _topk_indices(d, self.pool_size)
            sparse_top = _topk_indices(s, self.pool_size)
            cand = [int(c) for c in dict.fromkeys(list(dense_top) + list(sparse_top))]
            pools.append(cand)
            dense_pool.append(d[cand])
            sparse_pool.append(s[cand])

        return pools, dense_pool, sparse_pool, query_emb

    def retrieve(self, query: str) -> List[int]:
        """Retrieve candidate indices for a single query (union of dense+sparse).

        Kept for lighter, single-reranker use cases (see :mod:`occupations`); the
        full fusion path uses :meth:`pool` instead.
        """
        if self.corpus_embeddings is None:
            raise ValueError("Must call index() before retrieve().")

        query_emb = self.encode([query], show_progress=False)[0]
        query_sparse = self._tfidf_vectorizer.transform([query])
        d = self._dense_scores(query_emb)
        s = self._sparse_scores(query_sparse[0])
        dense_top = _topk_indices(d, self.pool_size)
        sparse_top = _topk_indices(s, self.pool_size)
        combined = list(dict.fromkeys(list(dense_top) + list(sparse_top)))
        return [int(i) for i in combined if 0 <= i < len(self._corpus_texts)]
