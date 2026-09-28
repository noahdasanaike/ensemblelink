"""
A TF-IDF vectorizer whose vocabulary pass runs in parallel and returns exactly
what scikit-learn's serial pass returns.

``TfidfVectorizer.fit_transform`` spends nearly all of its time in
``_count_vocab``, a pure-Python loop that assigns each new n-gram the next
integer id (first-appearance order) and builds a count matrix whose row entries
are sorted by that id. Everything downstream (feature sorting, idf, the l2 row
normalisation) is a deterministic function of that matrix, and the l2 norm sums
each row in stored order, so reproducing the matrix exactly reproduces every
float of the TF-IDF matrix.

The parallel pass counts contiguous chunks of documents in worker processes,
then rebuilds the serial first-appearance ids by walking the chunks in order
(a chunk's local ids are already in first-appearance order within that chunk),
remaps the column indices, and re-sorts each row by the global id. The result is
the same vocabulary and count matrix as the serial pass, bit for bit. Any
failure falls back to the serial pass.
"""

import os
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Below this many documents the process start-up costs more than it saves.
PARALLEL_MIN_DOCS = 100_000


def _count_chunk(params, docs):
    """Worker: serial ``_count_vocab`` on one chunk; returns terms in local-id order."""
    cv = CountVectorizer(**params)
    vocabulary, X = cv._count_vocab(docs, False)
    terms = [None] * len(vocabulary)
    for term, idx in vocabulary.items():
        terms[idx] = term
    return terms, X


def _n_jobs_default():
    n = os.cpu_count() or 1
    try:
        n = len(os.sched_getaffinity(0))  # respects SLURM/cgroup CPU limits
    except (AttributeError, OSError):
        pass
    return max(1, min(8, n))


class ParallelTfidfVectorizer(TfidfVectorizer):
    """``TfidfVectorizer`` with a parallel, exact vocabulary pass on large corpora.

    ``n_jobs`` worker processes (default: up to 8) are used when fitting on at
    least ``PARALLEL_MIN_DOCS`` documents; transforms and small fits are the
    unchanged scikit-learn code. The fitted object is an ordinary
    ``TfidfVectorizer`` in every other respect.
    """

    # Explicit signature: scikit-learn's parameter introspection rejects **kwargs.
    def __init__(self, *, analyzer="char", ngram_range=(2, 4), lowercase=True,
                 max_features=None, dtype=np.float64, n_jobs=None):
        super().__init__(analyzer=analyzer, ngram_range=ngram_range, lowercase=lowercase,
                         max_features=max_features, dtype=dtype)
        self.n_jobs = n_jobs

    def _count_vocab(self, raw_documents, fixed_vocab):
        if fixed_vocab:
            return super()._count_vocab(raw_documents, fixed_vocab)
        docs = raw_documents if isinstance(raw_documents, list) else list(raw_documents)
        n_jobs = self.n_jobs if self.n_jobs is not None else _n_jobs_default()
        if n_jobs <= 1 or len(docs) < PARALLEL_MIN_DOCS or not callable(getattr(CountVectorizer, "_count_vocab", None)):
            return super()._count_vocab(docs, fixed_vocab)
        try:
            return self._parallel_count_vocab(docs, n_jobs)
        except Exception:  # pragma: no cover - any worker problem: do it serially
            return super()._count_vocab(docs, fixed_vocab)

    def _parallel_count_vocab(self, docs, n_jobs):
        from joblib import Parallel, delayed

        # Only the parameters that the analyzer and the count matrix depend on.
        params = dict(
            input=self.input, encoding=self.encoding, decode_error=self.decode_error,
            strip_accents=self.strip_accents, lowercase=self.lowercase,
            preprocessor=self.preprocessor, tokenizer=self.tokenizer,
            stop_words=self.stop_words, token_pattern=self.token_pattern,
            ngram_range=self.ngram_range, analyzer=self.analyzer, dtype=self.dtype,
        )
        n_chunks = n_jobs * 4
        bounds = np.linspace(0, len(docs), n_chunks + 1).astype(int)
        chunks = [docs[bounds[i]:bounds[i + 1]] for i in range(n_chunks) if bounds[i + 1] > bounds[i]]
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_count_chunk)(params, c) for c in chunks
        )

        # Serial first-appearance ids: walk chunks in order, new terms get the next id.
        vocabulary = {}
        mats = []
        for terms, X in results:
            local_to_global = np.fromiter(
                (vocabulary.setdefault(t, len(vocabulary)) for t in terms),
                dtype=np.int64, count=len(terms),
            )
            X = X.tocsr()
            X.indices = local_to_global[X.indices]
            mats.append(X)

        n_features = len(vocabulary)
        indptr = [np.zeros(1, dtype=np.int64)]
        offset = 0
        for X in mats:
            indptr.append(X.indptr[1:].astype(np.int64) + offset)
            offset += X.indptr[-1]
        indptr = np.concatenate(indptr)
        indices = np.concatenate([X.indices for X in mats])
        data = np.concatenate([X.data for X in mats])
        # Same index dtype rule as scikit-learn's serial pass.
        idx_dtype = np.int64 if indptr[-1] > np.iinfo(np.int32).max else np.int32
        X = sp.csr_matrix(
            (data, indices.astype(idx_dtype), indptr.astype(idx_dtype)),
            shape=(len(docs), n_features), dtype=self.dtype,
        )
        X.sort_indices()
        return vocabulary, X
