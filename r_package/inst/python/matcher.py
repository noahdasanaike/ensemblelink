"""
Python backend of the ensemblelink R package.

The linkage code is the Python package ``zeroshot_linkage`` itself, vendored as
``ensemblelink_py`` next to this file (``tools/sync_r_backend.py`` copies it;
the test suite checks the copy), so R and Python give the same results. This
module only adapts it to the R interface: an index/match matcher over plain
character vectors and a blocked matcher.

Defaults are the paper's specification: harrier-oss-v1-0.6b embeddings, the
Jina v2 + BGE v2-m3 reranker expert, 30 candidates per retriever, bfloat16
inference on a GPU, and the B2 confidence (see the R documentation).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ensemblelink_py.core import FusionMatcher  # noqa: E402
from ensemblelink_py.fusion import MULTIFIELD_PROMPT  # noqa: E402
from ensemblelink_py.reranker import JINA_RERANKER, BGE_RERANKER  # noqa: E402

import numpy as np  # noqa: E402


def _device(device):
    return None if device in (None, "auto") else device


class EnsembleMatcher:
    """Four-expert agreement-fusion matcher over character vectors.

    Parameters mirror :class:`zeroshot_linkage.FusionMatcher`; ``reranker_model``,
    ``reranker_model_2`` and ``reranker_model_3`` form the reranker expert (None drops
    one). Default Jina v2 + BGE v2-m3; ``reranker_model=None,
    reranker_model_3="zeroentropy/zerank-2-reranker"`` is the commercial-use set.
    """

    def __init__(
        self,
        embedding_model="microsoft/harrier-oss-v1-0.6b",
        reranker_model=JINA_RERANKER,
        reranker_model_2=BGE_RERANKER,
        reranker_model_3=None,
        pool_size=30,
        max_length=1024,
        csls_k=10,
        device="auto",
        cache_dir=None,
        exact=True,
        index_cache=None,
        n_threads=None,
    ):
        names = [m for m in (reranker_model, reranker_model_2, reranker_model_3) if m]
        if not names:
            raise ValueError("At least one reranker is required")
        self._m = FusionMatcher(
            embedding_model=embedding_model, reranker_models=names, pool_size=int(pool_size),
            max_length=int(max_length), csls_k=int(csls_k), device=_device(device), cache_dir=cache_dir,
            exact=bool(exact), index_cache=index_cache, n_threads=None if n_threads is None else int(n_threads),
        )
        self._corpus = None
        self._batch_size = 50000

    def index(self, corpus, show_progress=True, batch_size=50000):
        """Index the corpus (or reuse / load an identical index)."""
        self._corpus = [str(c) for c in corpus]
        self._batch_size = int(batch_size)
        self._m.retriever.index(self._corpus, show_progress=show_progress, batch_size=self._batch_size)

    def match(self, queries, return_scores=False, show_progress=True, multifield=False):
        """Best corpus record per query.

        Returns the matched strings (None without candidates) or, with
        ``return_scores``, a dict with ``matches``, ``indices`` (0-based, None without
        candidates), ``score`` (B2 confidence), ``margin`` and ``reranker_probability``.
        ``multifield=True`` gives queries the multi-field embedding instruction.
        """
        if self._corpus is None:
            raise ValueError("Must call index() before match()")
        res = self._m.link([str(q) for q in queries], self._corpus, show_progress=show_progress,
                           batch_size=self._batch_size, query_prompt=MULTIFIELD_PROMPT if multifield else None)
        idx = [int(i) if i >= 0 else None for i in res["match_idx"]]
        matches = [self._corpus[i] if i is not None else None for i in idx]
        if not return_scores:
            return matches

        def clean(a):
            return [None if not np.isfinite(x) else float(x) for x in a]

        return {"matches": matches, "indices": idx, "score": clean(res["score"]), "margin": clean(res["margin"]),
                "reranker_probability": clean(res["reranker_probability"])}

    def match_one(self, query, return_score=False):
        res = self.match([query], return_scores=True, show_progress=False)
        return (res["matches"][0], res["score"][0]) if return_score else res["matches"][0]


class BlockedMatcher:
    """Hierarchical linkage: match blocks (e.g. states) first, then details within the matched block."""

    def __init__(self, **kwargs):
        self._matcher = EnsembleMatcher(**kwargs)
        self._corpus_blocks = None
        self._corpus_details = None
        self._block_to_rows = None
        self._unique_corpus_blocks = None

    def index(self, corpus_blocks, corpus_details, show_progress=True):
        corpus_blocks = [str(x) for x in corpus_blocks]
        corpus_details = [str(x) for x in corpus_details]
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
        query_blocks = [str(x) for x in query_blocks]
        query_details = [str(x) for x in query_details]
        if len(query_blocks) != len(query_details):
            raise ValueError("query_blocks and query_details must have same length")

        unique_query_blocks = list(dict.fromkeys(query_blocks))
        self._matcher.index(self._unique_corpus_blocks, show_progress=False)
        b = self._matcher.match(unique_query_blocks, return_scores=True, show_progress=show_progress)
        block_mapping = {qb: (mb, sc) for qb, mb, sc in zip(unique_query_blocks, b["matches"], b["score"])}

        n = len(query_blocks)
        match_blocks, match_details, match_indices = [None] * n, [None] * n, [None] * n
        block_scores, detail_scores = [None] * n, [None] * n
        grouped = {}
        for q_idx, qb in enumerate(query_blocks):
            mb, bsc = block_mapping.get(qb, (None, None))
            match_blocks[q_idx], block_scores[q_idx] = mb, bsc
            if mb is not None and mb in self._block_to_rows:
                grouped.setdefault(mb, []).append(q_idx)
        for mb, q_indices in grouped.items():
            rows = self._block_to_rows[mb]
            self._matcher.index([self._corpus_details[r] for r in rows], show_progress=False)
            d = self._matcher.match([query_details[q] for q in q_indices], return_scores=True, show_progress=False)
            for q_idx, local, md, dsc in zip(q_indices, d["indices"], d["matches"], d["score"]):
                match_indices[q_idx] = rows[local] if local is not None else None
                match_details[q_idx] = md
                detail_scores[q_idx] = dsc
        result = {"match_blocks": match_blocks, "match_details": match_details, "match_indices": match_indices}
        if return_scores:
            result["block_scores"] = block_scores
            result["detail_scores"] = detail_scores
        return result
