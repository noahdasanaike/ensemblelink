"""
The EnsembleLink record-linkage core: four experts fused without labels.

All queries are processed jointly so that the agreement weighting and the CSLS
hubness correction can look across the full query set. The pipeline:

  1. Retrieve a candidate pool per query: the union of the top dense
     (embedding-cosine) and top sparse (char-n-gram TF-IDF) corpus rows.
  2. Score every pooled (query, candidate) pair with four experts:
       - reranker ensemble: per-pool z-scores of each cross-encoder, summed
       - CSLS-corrected dense cosine: ``2 * cosine - hubness``
       - sparse TF-IDF cosine
       - Jaro-Winkler lexical similarity
  3. z-normalize each expert within each query's pool.
  4. Weight experts by squared agreement with the consensus pick (label-free).
  5. Fuse by weighted sum and take the argmax as the match.

No labeled pairs are used at any stage, which is what makes the method
zero-shot: the only supervision is the agreement among the experts themselves.
"""

import numpy as np
from typing import List, Optional, Sequence, Tuple

from .retrieval import EnsembleRetriever
from .reranker import RerankerEnsemble
from .fusion import zscore, agreement_weights, csls_hubness

DEFAULT_EMBEDDING_MODEL = "microsoft/harrier-oss-v1-0.6b"
DEFAULT_RERANKER_MODELS = (
    "jinaai/jina-reranker-v2-base-multilingual",
    "BAAI/bge-reranker-v2-m3",
)


def _load_jaro_winkler():
    """Return a Jaro-Winkler similarity function, or None if unavailable."""
    try:
        from rapidfuzz.distance import JaroWinkler

        return lambda a, b: JaroWinkler.similarity(a, b)
    except Exception:
        try:
            import jellyfish

            return lambda a, b: jellyfish.jaro_winkler_similarity(a, b)
        except Exception:
            return None


class FusionMatcher:
    """
    Four-expert agreement-fusion matcher.

    Holds the embedding, reranker, and lexical experts and links a set of
    queries to a corpus in one batched call. Models load lazily on first use, so
    constructing a matcher is cheap.

    Parameters
    ----------
    embedding_model : str
        Dense embedding model (harrier by default).
    reranker_models : sequence of str
        Cross-encoder rerankers; their z-scored sum is the reranker expert.
    pool_size : int
        Candidates retrieved from each of dense and sparse retrieval.
    max_length : int
        Maximum tokenized pair length for the rerankers.
    csls_k : int
        Neighbourhood size for the CSLS hubness correction.
    device : str, optional
        Device for inference ("cuda" or "cpu").
    cache_dir : str, optional
        Directory to download/cache models.
    """

    def __init__(
        self,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        reranker_models: Sequence[str] = DEFAULT_RERANKER_MODELS,
        pool_size: int = 50,
        max_length: int = 128,
        csls_k: int = 10,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        self.retriever = EnsembleRetriever(
            embedding_model=embedding_model,
            pool_size=pool_size,
            device=device,
            cache_dir=cache_dir,
        )
        self.reranker = RerankerEnsemble(
            model_names=list(reranker_models),
            max_length=max_length,
            device=device,
            cache_dir=cache_dir,
        )
        self.pool_size = pool_size
        self.csls_k = csls_k
        self._jw = _load_jaro_winkler()
        if self._jw is None:
            raise ImportError(
                "The Jaro-Winkler expert requires `rapidfuzz` (or `jellyfish`). "
                "Install it with `pip install rapidfuzz`."
            )

    def match(
        self,
        query_texts: Sequence[str],
        corpus_texts: Sequence[str],
        show_progress: bool = True,
        batch_size: int = 50000,
    ) -> List[Tuple[Optional[int], Optional[float]]]:
        """Link each query to its best corpus match.

        Returns a list of ``(corpus_index, fused_score)`` tuples, one per query;
        ``(None, None)`` when no candidate could be retrieved.
        """
        query_texts = [str(q) for q in query_texts]
        corpus_texts = [str(c) for c in corpus_texts]
        n_queries = len(query_texts)
        if n_queries == 0 or len(corpus_texts) == 0:
            return [(None, None)] * n_queries

        # 1. Retrieve pools and their dense/sparse scores.
        self.retriever.index(corpus_texts, show_progress=show_progress, batch_size=batch_size)
        pools, dense_pool, sparse_pool, query_emb = self.retriever.pool(
            query_texts, show_progress=show_progress
        )
        corpus_emb = self.retriever.corpus_embeddings

        # 2a. Reranker scores over every pooled pair (one flat batch).
        pairs, offsets = [], [0]
        for qi, cand in enumerate(pools):
            for c in cand:
                pairs.append([query_texts[qi], corpus_texts[c]])
            offsets.append(len(pairs))
        reranker_raw = self.reranker.score_pairs(pairs, show_progress=show_progress)

        # 2b. CSLS hubness term per pooled candidate (uses the whole query set).
        hub = csls_hubness(corpus_emb, query_emb, pools, k=self.csls_k)

        # 3. Build the per-query, z-normalized expert matrices.
        expert_mats: List[Optional[np.ndarray]] = []
        for qi, cand in enumerate(pools):
            if not cand:
                expert_mats.append(None)
                continue
            a, b = offsets[qi], offsets[qi + 1]

            reranker_z = np.zeros(len(cand))
            for raw in reranker_raw:
                reranker_z = reranker_z + zscore(raw[a:b])

            dense = np.asarray(dense_pool[qi], dtype=np.float64)
            hub_vec = np.array([hub[c] for c in cand], dtype=np.float64)
            csls_z = zscore(2.0 * dense - hub_vec)

            sparse_z = zscore(sparse_pool[qi])
            jw_z = zscore([self._jw(query_texts[qi], corpus_texts[c]) for c in cand])

            expert_mats.append(np.stack([reranker_z, csls_z, sparse_z, jw_z], axis=1))

        # 4. Label-free squared-agreement weights.
        weights = agreement_weights(expert_mats)

        # 5. Fuse and pick the argmax per query.
        results: List[Tuple[Optional[int], Optional[float]]] = []
        for qi, cand in enumerate(pools):
            F = expert_mats[qi]
            if F is None or not cand:
                results.append((None, None))
                continue
            fused = F @ weights
            best = int(np.argmax(fused))
            results.append((cand[best], float(fused[best])))
        return results
