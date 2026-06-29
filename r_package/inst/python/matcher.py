"""
Python backend for the ensemblelink R package.

Self-contained implementation of EnsembleLink's four-expert agreement fusion, so
the R package does not depend on the separately distributed ``zeroshot_linkage``
Python package. Candidates are retrieved by an embedding-plus-TF-IDF ensemble and
scored by four experts (a two-model reranker ensemble, a CSLS-corrected dense
cosine, a sparse TF-IDF cosine, and Jaro-Winkler similarity), then fused without
labels by weighting each expert by its agreement with the consensus pick.
"""

import numpy as np
from collections import Counter
from tqdm import tqdm
import torch


# ---------------------------------------------------------------------------
# Label-free fusion primitives
# ---------------------------------------------------------------------------

def _zscore(a):
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        return a
    s = a.std()
    return (a - a.mean()) / s if s > 1e-9 else np.zeros_like(a)


def _agreement_weights(expert_mats):
    mats = [F for F in expert_mats if F is not None and F.shape[0] > 0]
    if not mats:
        return None
    n_experts = mats[0].shape[1]
    agree = np.zeros(n_experts)
    for F in mats:
        picks = [int(np.argmax(F[:, e])) for e in range(n_experts)]
        consensus = Counter(picks).most_common(1)[0][0]
        for e in range(n_experts):
            agree[e] += int(picks[e] == consensus)
    w = (agree / len(mats)) ** 2
    if w.sum() <= 0:
        w = np.ones(n_experts)
    return w


def _topk(scores, k):
    n = scores.shape[0]
    k = min(k, n)
    if k <= 0:
        return np.array([], dtype=int)
    if k < n:
        part = np.argpartition(-scores, k - 1)[:k]
        return part[np.argsort(-scores[part])]
    return np.argsort(-scores)


def _load_jaro_winkler():
    try:
        from rapidfuzz.distance import JaroWinkler

        return lambda a, b: JaroWinkler.similarity(a, b)
    except Exception:
        try:
            import jellyfish

            return lambda a, b: jellyfish.jaro_winkler_similarity(a, b)
        except Exception:
            raise ImportError(
                "The Jaro-Winkler expert requires `rapidfuzz` (or `jellyfish`)."
            )


# ---------------------------------------------------------------------------
# Cross-encoder reranker (handles both Jina compute_score and CrossEncoder)
# ---------------------------------------------------------------------------

class _Reranker:
    def __init__(self, model_name, max_length=128, device="cpu", cache_dir=None):
        self.model_name = model_name
        self.max_length = max_length
        self.device = device
        self.cache_dir = cache_dir
        self._model = None
        self._kind = "jina" if "jina" in model_name.lower() else "cross-encoder"

    def _load(self):
        if self._model is not None:
            return
        if self._kind == "jina":
            from transformers import AutoModelForSequenceClassification

            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, torch_dtype="auto", trust_remote_code=True,
                cache_dir=self.cache_dir,
            )
            self._model.to(self.device)
            self._model.eval()
        else:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self.model_name, max_length=self.max_length, trust_remote_code=True,
                cache_folder=self.cache_dir, device=self.device,
            )

    def score_pairs(self, pairs):
        if not pairs:
            return np.array([])
        self._load()
        if self._kind == "jina":
            scores = self._model.compute_score(pairs, max_length=self.max_length)
        else:
            scores = self._model.predict(pairs, batch_size=256, show_progress_bar=False)
        # A single pair can come back as a Python float / 0-d array; force 1-d.
        return np.atleast_1d(np.asarray(scores, dtype=np.float64))


# ---------------------------------------------------------------------------
# Matcher
# ---------------------------------------------------------------------------

class EnsembleMatcher:
    """
    Zero-shot record linkage using four-expert agreement fusion.

    Parameters
    ----------
    embedding_model : str
        Dense embedding model (harrier by default).
    reranker_model, reranker_model_2 : str
        Cross-encoder rerankers forming the reranker expert. Set reranker_model_2
        to None to use a single reranker.
    pool_size : int
        Candidates retrieved from each of dense and sparse retrieval.
    max_length : int
        Maximum tokenized reranker pair length.
    csls_k : int
        Neighbourhood size for the CSLS hubness correction.
    device : str
        "cuda", "cpu", or "auto".
    """

    def __init__(
        self,
        embedding_model="microsoft/harrier-oss-v1-0.6b",
        reranker_model="jinaai/jina-reranker-v2-base-multilingual",
        reranker_model_2="BAAI/bge-reranker-v2-m3",
        pool_size=50,
        max_length=128,
        csls_k=10,
        device="auto",
        cache_dir=None,
    ):
        self.embedding_model_name = embedding_model
        self.reranker_model_names = [m for m in (reranker_model, reranker_model_2) if m]
        self.pool_size = int(pool_size)
        self.max_length = int(max_length)
        self.csls_k = int(csls_k)
        self.cache_dir = cache_dir
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device

        self._embedding_model = None
        self._rerankers = None
        self._jw = _load_jaro_winkler()

        # Corpus state
        self._corpus = None
        self._corpus_emb = None
        self._tfidf = None
        self._tfidf_T = None

    def _load_embedding_model(self):
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer

            self._embedding_model = SentenceTransformer(
                self.embedding_model_name, device=self.device,
                cache_folder=self.cache_dir, trust_remote_code=True,
            )

    def _load_rerankers(self):
        if self._rerankers is None:
            self._rerankers = [
                _Reranker(name, max_length=self.max_length, device=self.device,
                          cache_dir=self.cache_dir)
                for name in self.reranker_model_names
            ]

    def _encode(self, texts, show_progress=False, batch_size=50000):
        self._load_embedding_model()
        if len(texts) <= batch_size:
            return self._embedding_model.encode(
                texts, normalize_embeddings=True, show_progress_bar=show_progress,
                convert_to_numpy=True,
            ).astype(np.float32)
        first = self._embedding_model.encode(
            texts[:1], normalize_embeddings=True, convert_to_numpy=True
        )
        out = np.empty((len(texts), first.shape[1]), dtype=np.float32)
        out[0] = first[0]
        for start in range(1, len(texts), batch_size):
            end = min(start + batch_size, len(texts))
            out[start:end] = self._embedding_model.encode(
                texts[start:end], normalize_embeddings=True,
                show_progress_bar=show_progress, convert_to_numpy=True,
            ).astype(np.float32)
        return out

    def index(self, corpus, show_progress=True, batch_size=50000):
        """Build the dense and sparse indices for the corpus."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._corpus = list(corpus)
        self._corpus_emb = self._encode(self._corpus, show_progress=show_progress, batch_size=batch_size)
        self._tfidf = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), lowercase=True, max_features=50000)
        mat = self._tfidf.fit_transform(self._corpus)
        self._tfidf_T = mat.T.tocsr()

    def match(self, queries, return_scores=False, show_progress=True):
        """Match queries to corpus; returns matches (and scores if requested)."""
        if self._corpus is None:
            raise ValueError("Must call index() before match()")
        queries = [str(q) for q in queries]
        n_q = len(queries)
        if n_q == 0 or not self._corpus:
            empty = [self._corpus[0] if self._corpus else ""] * n_q
            return (empty, [0.0] * n_q) if return_scores else empty

        q_emb = self._encode(queries, show_progress=show_progress)
        q_sparse = self._tfidf.transform(queries)

        # 1. Pools + dense/sparse pooled scores.
        pools, dense_pool, sparse_pool = [], [], []
        for qi in range(n_q):
            d = self._corpus_emb @ q_emb[qi]
            s = (q_sparse[qi] @ self._tfidf_T).toarray().ravel()
            cand = [int(c) for c in dict.fromkeys(list(_topk(d, self.pool_size)) + list(_topk(s, self.pool_size)))]
            pools.append(cand)
            dense_pool.append(d[cand])
            sparse_pool.append(s[cand])

        # 2a. Reranker scores over pooled pairs.
        self._load_rerankers()
        pairs, offs = [], [0]
        for qi, cand in enumerate(pools):
            for c in cand:
                pairs.append([queries[qi], self._corpus[c]])
            offs.append(len(pairs))
        reranker_raw = [r.score_pairs(pairs) for r in self._rerankers]

        # 2b. CSLS hubness term.
        uniq = sorted({c for cand in pools for c in cand})
        hub = {}
        if uniq:
            sub = self._corpus_emb[uniq] @ q_emb.T
            kk = min(self.csls_k, sub.shape[1])
            hub = {uniq[i]: float(np.mean(np.sort(sub[i])[-kk:])) for i in range(len(uniq))}

        # 3. Per-query z-normalized expert matrices.
        expert_mats = []
        iterator = range(n_q)
        if show_progress:
            iterator = tqdm(iterator, desc="Matching")
        for qi in iterator:
            cand = pools[qi]
            if not cand:
                expert_mats.append(None)
                continue
            a, b = offs[qi], offs[qi + 1]
            reranker_z = np.zeros(len(cand))
            for raw in reranker_raw:
                reranker_z = reranker_z + _zscore(raw[a:b])
            dense = np.asarray(dense_pool[qi], dtype=np.float64)
            hub_vec = np.array([hub[c] for c in cand], dtype=np.float64)
            csls_z = _zscore(2.0 * dense - hub_vec)
            sparse_z = _zscore(sparse_pool[qi])
            jw_z = _zscore([self._jw(queries[qi], self._corpus[c]) for c in cand])
            expert_mats.append(np.stack([reranker_z, csls_z, sparse_z, jw_z], axis=1))

        # 4. Agreement weights, 5. fuse + argmax.
        weights = _agreement_weights(expert_mats)
        matches, scores = [], []
        for qi in range(n_q):
            F = expert_mats[qi]
            if F is None:
                matches.append(self._corpus[0] if self._corpus else "")
                scores.append(0.0)
                continue
            fused = F @ weights
            best = int(np.argmax(fused))
            matches.append(self._corpus[pools[qi][best]])
            scores.append(float(fused[best]))

        return (matches, scores) if return_scores else matches

    def match_one(self, query, return_score=False):
        """Match a single query (convenience wrapper around match)."""
        result = self.match([query], return_scores=True, show_progress=False)
        match, score = result[0][0], result[1][0]
        return (match, score) if return_score else match


class BlockedMatcher:
    """
    Hierarchical record linkage with blocking.

    Matches coarse blocking values first (e.g. states), then matches detail
    values within the matched block (e.g. counties). Both stages use the same
    four-expert agreement-fusion matcher.
    """

    def __init__(
        self,
        embedding_model="microsoft/harrier-oss-v1-0.6b",
        reranker_model="jinaai/jina-reranker-v2-base-multilingual",
        reranker_model_2="BAAI/bge-reranker-v2-m3",
        pool_size=50,
        max_length=128,
        csls_k=10,
        device="auto",
        cache_dir=None,
    ):
        self._kwargs = dict(
            embedding_model=embedding_model, reranker_model=reranker_model,
            reranker_model_2=reranker_model_2, pool_size=pool_size,
            max_length=max_length, csls_k=csls_k, device=device, cache_dir=cache_dir,
        )
        # One shared matcher reuses loaded models across blocks and details.
        self._matcher = EnsembleMatcher(**self._kwargs)
        self._corpus_blocks = None
        self._corpus_details = None
        self._block_to_rows = None
        self._unique_corpus_blocks = None

    def index(self, corpus_blocks, corpus_details, show_progress=True):
        corpus_blocks = list(corpus_blocks)
        corpus_details = list(corpus_details)
        if len(corpus_blocks) != len(corpus_details):
            raise ValueError("corpus_blocks and corpus_details must have same length")
        self._corpus_blocks = corpus_blocks
        self._corpus_details = corpus_details
        self._unique_corpus_blocks = list(dict.fromkeys(corpus_blocks))
        self._block_to_rows = {}
        for idx, block in enumerate(corpus_blocks):
            self._block_to_rows.setdefault(block, []).append(idx)

    def match(self, query_blocks, query_details, return_scores=False, show_progress=True):
        if self._corpus_blocks is None:
            raise ValueError("Must call index() before match()")
        query_blocks = list(query_blocks)
        query_details = list(query_details)
        if len(query_blocks) != len(query_details):
            raise ValueError("query_blocks and query_details must have same length")

        # Stage 1: match unique query blocks to unique corpus blocks.
        unique_query_blocks = list(dict.fromkeys(query_blocks))
        self._matcher.index(self._unique_corpus_blocks, show_progress=False)
        b_match, b_score = self._matcher.match(
            unique_query_blocks, return_scores=True, show_progress=show_progress
        )
        block_mapping = {
            qb: (mb, sc) for qb, mb, sc in zip(unique_query_blocks, b_match, b_score)
        }

        # Stage 2: detail matching within each matched block, grouped.
        n = len(query_blocks)
        match_blocks = [None] * n
        match_details = [None] * n
        match_indices = [None] * n
        block_scores = [None] * n
        detail_scores = [None] * n

        grouped = {}
        for q_idx, qb in enumerate(query_blocks):
            mb, bsc = block_mapping.get(qb, (None, None))
            match_blocks[q_idx] = mb
            block_scores[q_idx] = bsc
            if mb is not None and mb in self._block_to_rows:
                grouped.setdefault(mb, []).append(q_idx)

        for mb, q_indices in grouped.items():
            rows = self._block_to_rows[mb]
            sub_corpus = [self._corpus_details[r] for r in rows]
            sub_queries = [query_details[q] for q in q_indices]
            self._matcher.index(sub_corpus, show_progress=False)
            d_match, d_score = self._matcher.match(
                sub_queries, return_scores=True, show_progress=False
            )
            for q_idx, md, dsc in zip(q_indices, d_match, d_score):
                # Recover the global corpus index from the matched detail string.
                try:
                    local = sub_corpus.index(md)
                    match_indices[q_idx] = rows[local]
                except ValueError:
                    match_indices[q_idx] = None
                match_details[q_idx] = md
                detail_scores[q_idx] = dsc

        result = {
            "match_blocks": match_blocks,
            "match_details": match_details,
            "match_indices": match_indices,
        }
        if return_scores:
            result["block_scores"] = block_scores
            result["detail_scores"] = detail_scores
        return result
