"""
Candidate retrieval: exact dense search plus character n-gram TF-IDF.

Dense retrieval embeds records with a sentence-transformer (harrier-oss-v1-0.6b
by default) and ranks the whole corpus by cosine; sparse retrieval ranks it by
the cosine of character 2-4-gram TF-IDF vectors of transliterated (Unidecode)
text. A query's candidate pool is the union of the top ``pool_size`` records of
each. When a query shares no character n-gram with any record, the pool takes
the top ``2 * pool_size`` dense records instead.

Numerics follow the paper's benchmark engine:

* the embedding model runs in bfloat16 with SDPA attention on a GPU (float32 on
  CPU), at most ``embed_max_length`` tokens per record, and embeddings are
  L2-normalized and stored in float32;
* the TF-IDF matrix is float32;
* dense and sparse scores are computed on the GPU, 16 queries at a time
  (float32, no TF32; the TF-IDF product with cuSPARSE), and the top-k is taken
  on the CPU with ties broken by corpus row order. cuSPARSE sums a row in an
  order of its own, so two records whose TF-IDF cosines are equal in exact
  arithmetic usually get slightly different float scores (as in the benchmark),
  and its sums can vary in the last bit between runs; without a GPU the
  product runs on the CPU, where such exact ties are kept;
* the CSLS hubness of a record is the mean of its 10 highest cosines with the
  query set, computed on the GPU in blocks of 8,192 records.

Two batching modes (``exact``):

* ``exact=True`` (default) feeds the models exactly as the benchmark does:
  every text in input order, in calls of 10,000 texts at batch size 256.
* ``exact=False`` embeds each distinct text once, longest first, in batches
  sized to the text length, and scores 256 queries per GPU product. Faster,
  above all with duplicated records; bfloat16 embeddings then differ at noise
  level, which can reorder nearly tied candidates.
"""

import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .fusion import lexical_text, top_k
from .reranker import model_revision, resolve_device, resolve_dtype, MODEL_REVISIONS

EXACT_EMBED_CHUNK = 10000   # texts per encode call in exact mode (the benchmark's chunk)
DEFAULT_EMBED_BATCH = 256
DEFAULT_EMBED_MAX_LENGTH = 512
EXACT_QUERY_BLOCK = 16      # queries per GPU dense/sparse product in exact mode
FAST_QUERY_BLOCK = 256
HUB_BLOCK = 8192
# Adaptive embedding batches (exact=False): about this many characters per batch,
# never fewer than 32 texts and never more than 256.
EMBED_CHAR_BUDGET = 32768
EMBED_MIN_BATCH = 32
EMBED_MAX_BATCH = 256


def lexical_texts(texts: Sequence[str]) -> List[str]:
    """``lexical_text`` over a list, transliterating each distinct string once."""
    memo: Dict[str, str] = {}
    out = []
    for t in texts:
        v = memo.get(t)
        if v is None:
            v = lexical_text(t)
            memo[t] = v
        out.append(v)
    return out


def unique_inverse(items: Sequence) -> Tuple[list, np.ndarray]:
    """Distinct items in first-occurrence order, and each item's position in that list."""
    pos: dict = {}
    inverse = np.empty(len(items), dtype=np.int64)
    for i, x in enumerate(items):
        j = pos.get(x)
        if j is None:
            j = len(pos)
            pos[x] = j
        inverse[i] = j
    return list(pos.keys()), inverse


class EnsembleRetriever:
    """
    Dense plus sparse candidate retrieval over a corpus.

    Parameters
    ----------
    embedding_model : str
        Sentence-transformer model (default harrier-oss-v1-0.6b, pinned revision).
    pool_size : int
        Candidates from each of dense and sparse retrieval (default 30). ``top_k`` is an alias.
    ngram_range : tuple
        Character n-gram range of the TF-IDF index (default (2, 4)).
    max_features : int, optional
        TF-IDF vocabulary cap (default none: every observed n-gram).
    device : str, optional
        "cuda", "cpu" or None/"auto".
    cache_dir : str, optional
        Model cache directory.
    exact : bool
        Benchmark batching (True, default) or deduplicated, length-sorted batching (False).
    embed_batch_size : int, optional
        Texts per embedding batch (exact default 256; fast default adaptive).
    index_cache : str, optional
        Directory for the on-disk corpus-index cache (see :mod:`index_cache`).
    n_threads : int, optional
        Worker processes for the TF-IDF vocabulary pass (default up to 8).
    embed_max_length : int
        Token ceiling per record for the embedding model (default 512).
    dtype : str or torch.dtype
        "auto" (default): bfloat16 on GPU, float32 on CPU.
    """

    def __init__(
        self,
        embedding_model: str = "microsoft/harrier-oss-v1-0.6b",
        pool_size: Optional[int] = None,
        top_k: Optional[int] = None,
        ngram_range: tuple = (2, 4),
        max_features: Optional[int] = None,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        exact: bool = True,
        embed_batch_size: Optional[int] = None,
        index_cache: Optional[str] = None,
        n_threads: Optional[int] = None,
        embed_max_length: int = DEFAULT_EMBED_MAX_LENGTH,
        dtype="auto",
    ):
        self.embedding_model_name = embedding_model
        self.pool_size = pool_size if pool_size is not None else (top_k if top_k is not None else 30)
        self.top_k = self.pool_size
        self.ngram_range = tuple(ngram_range)
        self.max_features = max_features
        self.device = device
        self.cache_dir = cache_dir
        self.exact = exact
        self.embed_batch_size = embed_batch_size
        self.index_cache = index_cache
        self.n_threads = n_threads
        self.embed_max_length = int(embed_max_length)
        self.dtype = dtype

        self._embed_model = None
        self._tfidf_vectorizer = None
        self._tfidf_matrix = None
        self._gpu_index = None
        self.corpus_embeddings: Optional[np.ndarray] = None
        self._corpus_texts: List[str] = []
        self._index_settings = None
        # Seconds per index sub-stage of the last index() call, plus how it was served.
        self.last_index_timings: Dict[str, float] = {}

    # ------------------------------------------------------------------ models

    def _on_gpu(self) -> bool:
        import torch

        self.device = resolve_device(self.device)
        return str(self.device).startswith("cuda") and torch.cuda.is_available()

    def _load_embedding_model(self):
        """Lazy-load the embedding model (bfloat16 + SDPA on GPU, as in the benchmark)."""
        if self._embed_model is not None:
            return
        from sentence_transformers import SentenceTransformer

        self.device = resolve_device(self.device)
        torch_dtype = resolve_dtype(self.dtype, self.device)
        name = self.embedding_model_name
        pinned = name in MODEL_REVISIONS
        kwargs = dict(device=self.device, cache_folder=self.cache_dir, revision=model_revision(name),
                      trust_remote_code=not pinned)
        try:
            model = SentenceTransformer(name, model_kwargs={"torch_dtype": torch_dtype, "attn_implementation": "sdpa"},
                                        **kwargs)
        except (TypeError, ValueError):
            if pinned:
                raise
            model = SentenceTransformer(name, model_kwargs={"torch_dtype": torch_dtype}, **kwargs)
        model.max_seq_length = self.embed_max_length
        self._embed_model = model

    # ------------------------------------------------------------------ encoding

    def _embedding_dimension(self) -> int:
        m = self._embed_model
        f = getattr(m, "get_embedding_dimension", None) or m.get_sentence_embedding_dimension
        return f()

    def encode(self, texts: List[str], show_progress: bool = True, batch_size: int = 50000,
               prompt: Optional[str] = None) -> np.ndarray:
        """L2-normalized float32 embeddings (queries may carry an instruction ``prompt``)."""
        if self.exact:
            return self._encode_exact(texts, show_progress, prompt)
        return self._encode_fast(texts, show_progress, batch_size, prompt)

    @staticmethod
    def _normalize(values: np.ndarray) -> np.ndarray:
        values = values.astype(np.float32)
        values /= np.linalg.norm(values, axis=1, keepdims=True)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite embedding (empty record text?)")
        return values

    def _encode_exact(self, texts, show_progress, prompt):
        """The benchmark's encoding: calls of 10,000 texts in input order at batch size 256."""
        self._load_embedding_model()
        texts = list(texts)
        batch = int(self.embed_batch_size or DEFAULT_EMBED_BATCH)
        dim = self._embedding_dimension()
        out = np.empty((len(texts), dim), dtype=np.float32)
        pbar = None
        if show_progress and len(texts) > EXACT_EMBED_CHUNK:
            from tqdm import tqdm

            pbar = tqdm(total=len(texts), desc="Embedding")
        for start in range(0, len(texts), EXACT_EMBED_CHUNK):
            chunk = texts[start:start + EXACT_EMBED_CHUNK]
            values = self._embed_model.encode(chunk, prompt=prompt or "", batch_size=batch, normalize_embeddings=True,
                                              convert_to_numpy=True,
                                              show_progress_bar=show_progress and pbar is None)
            out[start:start + len(chunk)] = self._normalize(values)
            if pbar is not None:
                pbar.update(len(chunk))
        if pbar is not None:
            pbar.close()
        return out

    def _batch_size_for(self, n_chars: int) -> int:
        if self.embed_batch_size is not None:
            return int(self.embed_batch_size)
        return int(np.clip(EMBED_CHAR_BUDGET // max(1, n_chars), EMBED_MIN_BATCH, EMBED_MAX_BATCH))

    def _encode_fast(self, texts, show_progress, batch_size, prompt):
        """Embed each distinct text once, longest first, in batches sized to the text length."""
        self._load_embedding_model()
        uniq, inverse = unique_inverse(list(texts))
        n = len(uniq)
        dim = self._embedding_dimension()
        if n == 0:
            return np.empty((len(texts), dim or 0), dtype=np.float32)
        plen = len(prompt) if prompt else 0
        lengths = np.fromiter((len(t) + plen for t in uniq), dtype=np.int64, count=n)
        order = np.argsort(-lengths, kind="stable")
        bs = np.array([self._batch_size_for(int(x)) for x in lengths[order]], dtype=np.int64)
        groups = []
        start = 0
        cap = max(1, int(batch_size))
        while start < n:
            end = start + 1
            while end < n and bs[end] == bs[start] and end - start < cap:
                end += 1
            groups.append((start, end, int(bs[start])))
            start = end
        pbar = None
        if show_progress:
            from tqdm import tqdm

            pbar = tqdm(total=n, desc="Embedding")
        out = np.empty((n, dim), dtype=np.float32)
        for a, b, size in groups:
            idx = order[a:b]
            emb = self._embed_model.encode([uniq[i] for i in idx], prompt=prompt or "", batch_size=size,
                                           normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
            out[idx] = self._normalize(emb)
            if pbar is not None:
                pbar.update(b - a)
        if pbar is not None:
            pbar.close()
        return out if n == len(texts) else out[inverse]

    # ------------------------------------------------------------------ indexing

    def _settings(self):
        return (self.embedding_model_name, model_revision(self.embedding_model_name), bool(self.exact),
                self.embed_batch_size, self.embed_max_length, str(self.dtype), tuple(self.ngram_range),
                self.max_features, str(resolve_device(self.device)))

    def _cache_components(self, texts) -> dict:
        from . import index_cache

        name = self.embedding_model_name
        self.device = resolve_device(self.device)
        return {
            "corpus_sha256": index_cache.hash_texts(texts),
            "n_records": len(texts),
            "embedding_model": name,
            "embedding_revision": model_revision(name) or index_cache.resolve_model_revision(name, self.cache_dir),
            "exact": bool(self.exact),
            "embed_batching": (
                {"chunk": EXACT_EMBED_CHUNK, "batch": int(self.embed_batch_size or DEFAULT_EMBED_BATCH)} if self.exact else
                {"fixed": self.embed_batch_size, "char_budget": EMBED_CHAR_BUDGET,
                 "min": EMBED_MIN_BATCH, "max": EMBED_MAX_BATCH}
            ),
            "embed_max_length": self.embed_max_length,
            "dtype": str(resolve_dtype(self.dtype, self.device)),
            "tfidf": {"analyzer": "char", "ngram_range": list(self.ngram_range), "lowercase": True,
                      "max_features": self.max_features, "lexical": "unidecode", "dtype": "float32"},
            "versions": index_cache.library_versions(),
            "device": index_cache.device_tag(self.device),
        }

    def index(self, texts: List[str], show_progress: bool = True, batch_size: int = 50000) -> None:
        """Build the dense and sparse indices of the corpus.

        ``batch_size`` caps the texts per embedding call in fast mode (exact mode
        always uses calls of 10,000). A corpus identical to the one already indexed
        is not re-indexed; with ``index_cache`` set, the index is loaded from or saved
        to disk.
        """
        texts = list(texts)
        settings = self._settings()
        timings: Dict[str, float] = {}
        if (self.corpus_embeddings is not None and self._index_settings == settings
                and len(texts) == len(self._corpus_texts) and texts == self._corpus_texts):
            self.last_index_timings = {"source": "memory"}
            return
        self._gpu_index = None

        key = components = None
        if self.index_cache:
            from . import index_cache

            t0 = time.perf_counter()
            components = self._cache_components(texts)
            key = index_cache.make_key(components)
            hit = index_cache.load(self.index_cache, key)
            timings["cache_key_and_lookup"] = time.perf_counter() - t0
            if hit is not None:
                self.corpus_embeddings, self._tfidf_matrix, self._tfidf_vectorizer = hit
                self._corpus_texts = texts
                self._index_settings = settings
                self.last_index_timings = {"source": "disk", "cache_key": key, **timings}
                return

        from ._fast_tfidf import ParallelTfidfVectorizer

        t0 = time.perf_counter()
        was_loaded = self._embed_model is not None
        self._load_embedding_model()
        timings["model_load"] = 0.0 if was_loaded else time.perf_counter() - t0
        t0 = time.perf_counter()
        emb = self.encode(texts, show_progress=show_progress, batch_size=batch_size)
        timings["embed"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        lex = lexical_texts(texts)
        timings["lexical"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        vec = ParallelTfidfVectorizer(analyzer="char", ngram_range=self.ngram_range, lowercase=True,
                                      max_features=self.max_features, dtype=np.float32, n_jobs=self.n_threads)
        mat = vec.fit_transform(lex)
        timings["tfidf_fit_transform"] = time.perf_counter() - t0

        self._corpus_texts = texts
        self.corpus_embeddings = emb
        self._tfidf_vectorizer = vec
        self._tfidf_matrix = mat
        self._index_settings = settings
        if self.index_cache:
            from . import index_cache

            t0 = time.perf_counter()
            index_cache.save(self.index_cache, key, components, emb, mat, vec)
            timings["cache_save"] = time.perf_counter() - t0
            timings["cache_key"] = key
        self.last_index_timings = {"source": "computed", **timings}

    # ------------------------------------------------------------------ scoring

    def _gpu(self):
        """Corpus embeddings and TF-IDF matrix (CSR) on the GPU, built once per index."""
        if self._gpu_index is None:
            import warnings

            import torch

            m = self._tfidf_matrix
            with warnings.catch_warnings():  # sparse CSR tensors are flagged as beta
                warnings.simplefilter("ignore", UserWarning)
                sparse = torch.sparse_csr_tensor(torch.from_numpy(m.indptr), torch.from_numpy(m.indices),
                                                 torch.from_numpy(m.data), size=m.shape, device=self.device)
            self._gpu_index = (
                torch.from_numpy(np.ascontiguousarray(self.corpus_embeddings)).to(self.device),
                sparse,
            )
        return self._gpu_index

    def _gpu_corpus(self):
        return self._gpu()[0]

    def release_gpu(self):
        """Free the GPU copy of the index (rebuilt on the next retrieval)."""
        self._gpu_index = None
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    def _score_block(self, q_emb_block: np.ndarray, q_sparse_block):
        """Dense and sparse scores of a block of queries against the whole corpus (float32).

        On a GPU, as in the benchmark: corpus embeddings times the block's query embeddings
        (cuBLAS), and the corpus TF-IDF matrix times the block's dense query columns
        (cuSPARSE). On CPU: the same products with NumPy and SciPy.
        """
        qd = np.ascontiguousarray(q_sparse_block.toarray().T, dtype=np.float32)
        if self._on_gpu():
            import torch

            corpus_gpu, sparse_gpu = self._gpu()
            with torch.inference_mode():
                qg = torch.from_numpy(np.ascontiguousarray(q_emb_block)).to(self.device)
                dense = (corpus_gpu @ qg.T).T.contiguous().cpu().numpy()
                sparse = torch.sparse.mm(sparse_gpu, torch.from_numpy(qd).to(self.device)).T.contiguous().cpu().numpy()
            return dense, sparse
        dense = q_emb_block @ self.corpus_embeddings.T
        sparse = np.ascontiguousarray((self._tfidf_matrix @ qd).T, dtype=np.float32)
        return dense, sparse

    def _pool_one(self, dense: np.ndarray, sparse: np.ndarray):
        k = self.pool_size
        di = top_k(dense, k)
        si = top_k(sparse, k)
        if not np.any(sparse > 0):
            # Dense refill: no shared character n-gram with any record, so the sparse top-k
            # would be arbitrary; take twice as many dense candidates instead.
            di = top_k(dense, 2 * k)
            si = np.array([], dtype=np.int64)
        cand = np.array(sorted(set(di.tolist()) | set(si.tolist())), dtype=np.int64)
        return cand, dense[cand], sparse[cand]

    def pool(self, query_texts: List[str], show_progress: bool = True, query_prompt: Optional[str] = None):
        """Candidate pools and pooled scores.

        Returns ``(pools, dense_pool, sparse_pool, query_emb)``: per query, the candidate
        corpus rows in ascending order, their dense cosines and TF-IDF cosines (float32),
        and the query embeddings.
        """
        if self.corpus_embeddings is None:
            raise ValueError("Must call index() before pool().")
        query_texts = list(query_texts)
        n = len(query_texts)
        query_emb = self.encode(query_texts, show_progress=show_progress, prompt=query_prompt)
        pools: List = [None] * n
        dense_pool: List = [None] * n
        sparse_pool: List = [None] * n
        if n == 0:
            return pools, dense_pool, sparse_pool, query_emb
        query_sparse = self._tfidf_vectorizer.transform(lexical_texts(query_texts)).astype(np.float32)
        block = EXACT_QUERY_BLOCK if self.exact else FAST_QUERY_BLOCK
        pbar = None
        if show_progress:
            from tqdm import tqdm

            pbar = tqdm(total=n, desc="Retrieving candidates")
        for b0 in range(0, n, block):
            b1 = min(b0 + block, n)
            dense, sparse = self._score_block(query_emb[b0:b1], query_sparse[b0:b1])
            for i in range(b1 - b0):
                pools[b0 + i], dense_pool[b0 + i], sparse_pool[b0 + i] = self._pool_one(dense[i], sparse[i])
            if pbar is not None:
                pbar.update(b1 - b0)
        if pbar is not None:
            pbar.close()
        return pools, dense_pool, sparse_pool, query_emb

    def hubness(self, rows: Sequence[int], query_emb: np.ndarray, k: int = 10) -> Dict[int, float]:
        """CSLS hubness of corpus rows: mean of each row's ``k`` highest cosines with the query set (float32).

        Computed on the GPU in blocks of 8,192 corpus rows. In exact mode every corpus row is
        scored, block by block in corpus order, as the benchmark does (a row's value then does
        not depend on which rows were pooled); in fast mode only the pooled rows.
        """
        rows = np.asarray(sorted({int(r) for r in rows}), dtype=np.int64)
        if len(rows) == 0:
            return {}
        kk = min(k, len(query_emb))
        n_corpus = self.corpus_embeddings.shape[0]
        scored = np.arange(n_corpus) if self.exact else rows
        out = np.empty(len(scored), dtype=np.float32)
        if self._on_gpu():
            import torch

            corpus_gpu = self._gpu_corpus()
            with torch.inference_mode():
                qg = torch.from_numpy(np.ascontiguousarray(query_emb)).to(self.device)
                for s in range(0, len(scored), HUB_BLOCK):
                    if self.exact:
                        block = corpus_gpu[s:s + HUB_BLOCK]
                    else:
                        block = corpus_gpu[torch.from_numpy(scored[s:s + HUB_BLOCK]).to(self.device)]
                    out[s:s + len(block)] = (block @ qg.T).topk(kk, dim=1).values.mean(dim=1).cpu().numpy()
        else:
            for s in range(0, len(scored), HUB_BLOCK):
                sims = self.corpus_embeddings[scored[s:s + HUB_BLOCK]] @ query_emb.T
                out[s:s + len(sims)] = np.sort(sims, axis=1)[:, -kk:].mean(axis=1)
        values = out[rows] if self.exact else out
        return dict(zip(rows.tolist(), values.tolist()))

    def retrieve(self, query: str) -> List[int]:
        """Candidate rows for a single query (union of dense and sparse top-k; used by :mod:`occupations`)."""
        pools = self.pool([query], show_progress=False)[0]
        return [int(i) for i in pools[0]]
