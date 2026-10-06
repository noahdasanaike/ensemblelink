"""
The EnsembleLink record-linkage core: four experts fused without labels.

All queries of a call are processed jointly: the CSLS hubness term, the
agreement weights and the confidence ranks look across the whole query set.

  1. Retrieve a candidate pool per query: the union of the top dense and top
     sparse (character TF-IDF) corpus rows; 2k dense rows when the query shares
     no character n-gram with the corpus.
  2. Score every pooled pair with four experts: the reranker ensemble
     (z-scored sum of Jina v2 and BGE v2-m3 probabilities), the CSLS-corrected
     dense cosine ``2 cos - hub``, the TF-IDF cosine, and Jaro-Winkler on
     transliterated text.
  3. z-score each expert within each pool; experts without spread abstain.
  4. Weight experts by their squared agreement with the consensus pick.
  5. The top-1 is the fused argmax; the confidence is rule B2 (see
     :func:`zeroshot_linkage.fusion.b2_confidence`).

No labeled pairs are used at any stage.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .fusion import (b2_confidence, expert_features, fuse, lexical_text, mean_probability, select_top)
from .reranker import (BGE_RERANKER, DEFAULT_RERANKER_BATCH, DEFAULT_RERANKER_MAX_LENGTH, JINA_RERANKER,
                       ZERANK_RERANKER, RerankerEnsemble)
from .retrieval import DEFAULT_EMBED_MAX_LENGTH, EnsembleRetriever, unique_inverse

DEFAULT_EMBEDDING_MODEL = "microsoft/harrier-oss-v1-0.6b"
# Default reranker expert: Jina Reranker v2 (CC-BY-NC-4.0, non-commercial) and BGE Reranker
# v2-m3 (Apache-2.0).
DEFAULT_RERANKER_MODELS = (JINA_RERANKER, BGE_RERANKER)
# Every model licensed for commercial use (Apache-2.0 / MIT): BGE v2-m3 and zerank-2 in place of Jina v2.
COMMERCIAL_RERANKER_MODELS = (BGE_RERANKER, ZERANK_RERANKER)


def _jaro_winkler():
    from rapidfuzz.distance import JaroWinkler

    return JaroWinkler.normalized_similarity


class FusionMatcher:
    """
    Four-expert agreement-fusion matcher. Models load lazily on first use.

    Parameters
    ----------
    embedding_model : str
        Dense embedding model (harrier-oss-v1-0.6b).
    reranker_models : sequence of str
        Rerankers forming the reranker expert (default Jina v2 and BGE v2-m3;
        ``COMMERCIAL_RERANKER_MODELS`` = BGE v2-m3 and zerank-2).
    pool_size : int
        Candidates from each of dense and sparse retrieval (default 30).
    max_length : int
        Token ceiling per reranker pair (default 1024; pairs below it are unaffected).
    embed_max_length : int
        Token ceiling per record for the embedding model (default 512).
    csls_k : int
        Neighbourhood size of the CSLS hubness term (default 10).
    device : str, optional
        "cuda" or "cpu" (default: GPU if available).
    cache_dir : str, optional
        Model cache directory.
    exact : bool
        True (default): the models see fixed batches, so results are
        deterministic for a given GPU up to GPU nondeterminism.
        False: each distinct query text, corpus text and (query, candidate) pair
        goes through the models once, in length-sorted batches; faster, above all
        with duplicated records, with bfloat16 scores that move at noise level.
    index_cache : str, optional
        Directory for an on-disk cache of the corpus index (embeddings, TF-IDF).
    embed_batch_size : int, optional
        Texts per embedding batch (exact default 256; fast default adaptive).
    reranker_batch_size : int
        Pairs per reranker batch (default 128).
    dtype : str
        Model precision: "auto" (default) = bfloat16 on GPU, float32 on CPU.
    n_threads : int, optional
        Worker processes for the TF-IDF vocabulary pass (default up to 8).
    """

    def __init__(
        self,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        reranker_models: Sequence[str] = DEFAULT_RERANKER_MODELS,
        pool_size: int = 30,
        max_length: int = DEFAULT_RERANKER_MAX_LENGTH,
        csls_k: int = 10,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        exact: bool = True,
        index_cache: Optional[str] = None,
        embed_batch_size: Optional[int] = None,
        n_threads: Optional[int] = None,
        reranker_batch_size: int = DEFAULT_RERANKER_BATCH,
        embed_max_length: int = DEFAULT_EMBED_MAX_LENGTH,
        dtype: str = "auto",
    ):
        self.retriever = EnsembleRetriever(
            embedding_model=embedding_model, pool_size=pool_size, device=device, cache_dir=cache_dir,
            exact=exact, embed_batch_size=embed_batch_size, index_cache=index_cache, n_threads=n_threads,
            embed_max_length=embed_max_length, dtype=dtype,
        )
        self.reranker = RerankerEnsemble(
            model_names=list(reranker_models), max_length=max_length, device=device, cache_dir=cache_dir,
            batch_size=reranker_batch_size, dtype=dtype,
        )
        self.pool_size = pool_size
        self.csls_k = csls_k
        self.exact = exact
        self._jw = _jaro_winkler()
        self.last_weights = None
        self.last_agreement = None

    def link(
        self,
        query_texts: Sequence[str],
        corpus_texts: Sequence[str],
        query_ids: Optional[Sequence[str]] = None,
        corpus_ids: Optional[Sequence[str]] = None,
        show_progress: bool = True,
        batch_size: int = 50000,
        query_prompt: Optional[str] = None,
        details: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Link each query to its best corpus record.

        ``query_ids`` / ``corpus_ids`` (unique strings; default the row positions) only
        break exact ties, by the smallest SHA-256 of ``"<query id>|<record id>"``.

        Returns a dict of per-query arrays: ``match_idx`` (corpus row, -1 without
        candidates), ``score`` (the B2 confidence, NaN without candidates),
        ``margin`` (top minus runner-up fused score; 0 for a tie at the top, NaN for a
        single candidate), ``reranker_probability`` (mean 0-1 reranker score of the
        match), ``fused_score`` (top fused score) and ``top_tie_count``. ``details=True``
        adds intermediate values (pools, pooled scores, hubness, reranker scores, fused
        scores, agreement weights) for inspection.
        """
        query_texts = [str(q) for q in query_texts]
        corpus_texts = [str(c) for c in corpus_texts]
        n = len(query_texts)
        qids = [str(i) for i in range(n)] if query_ids is None else [str(x) for x in query_ids]
        cids = [str(i) for i in range(len(corpus_texts))] if corpus_ids is None else [str(x) for x in corpus_ids]
        if len(qids) != n or len(cids) != len(corpus_texts):
            raise ValueError("IDs and texts differ in length")
        if len(set(qids)) != n or len(set(cids)) != len(cids):
            raise ValueError("Query and corpus IDs must be unique")
        out = {
            "match_idx": np.full(n, -1, dtype=np.int64),
            "score": np.full(n, np.nan),
            "margin": np.full(n, np.nan),
            "reranker_probability": np.full(n, np.nan),
            "fused_score": np.full(n, np.nan),
            "top_tie_count": np.zeros(n, dtype=np.int64),
        }
        if n == 0 or not corpus_texts:
            return out

        # 1. Pools and their dense/sparse scores (one pool per distinct text in fast mode).
        self.retriever.index(corpus_texts, show_progress=show_progress, batch_size=batch_size)
        if self.exact:
            uq_texts, q_inv = query_texts, np.arange(n)
        else:
            uq_texts, q_inv = unique_inverse(query_texts)
        pools, dense_pool, sparse_pool, uq_emb = self.retriever.pool(uq_texts, show_progress=show_progress,
                                                                     query_prompt=query_prompt)
        query_emb = uq_emb if len(uq_texts) == n else uq_emb[q_inv]   # every occurrence counts in CSLS

        # 2a. Reranker probabilities of the pooled pairs.
        if self.exact:
            pairs, offsets = [], [0]
            for qi, cand in enumerate(pools):
                pairs.extend([uq_texts[qi], corpus_texts[c]] for c in cand)
                offsets.append(len(pairs))
            pair_sel = [np.arange(offsets[i], offsets[i + 1]) for i in range(len(pools))]
            raw = self.reranker.score_pairs(pairs, show_progress=show_progress)
        else:
            pair_id, pairs, pair_sel = {}, [], []
            for qi, cand in enumerate(pools):
                q = uq_texts[qi]
                sel = np.empty(len(cand), dtype=np.int64)
                for j, c in enumerate(cand):
                    key = (q, corpus_texts[c])
                    p = pair_id.get(key)
                    if p is None:
                        p = pair_id[key] = len(pairs)
                        pairs.append([q, key[1]])
                    sel[j] = p
                pair_sel.append(sel)
            raw = self.reranker.score_pairs(pairs, show_progress=show_progress, sort_by_length=True)

        # 2b. CSLS hubness of every pooled record over the whole query set.
        hub = self.retriever.hubness([c for cand in pools for c in cand], query_emb, k=self.csls_k)

        # 3. Expert matrices per distinct pool.
        lex: dict = {}
        jw_memo: dict = {}

        def lexical(t):
            v = lex.get(t)
            if v is None:
                v = lex[t] = lexical_text(t)
            return v

        def jw(q, c):
            v = jw_memo.get((q, c))
            if v is None:
                v = jw_memo[(q, c)] = float(self._jw(lexical(q), lexical(c)))
            return v

        feats: List[Optional[np.ndarray]] = []
        for qi, cand in enumerate(pools):
            if len(cand) == 0:
                feats.append(None)
                continue
            sel = pair_sel[qi]
            dense = np.array([float(x) for x in dense_pool[qi]])
            hubv = np.array([hub[int(c)] for c in cand])
            sparse = np.array([float(x) for x in sparse_pool[qi]])
            jws = np.array([jw(uq_texts[qi], corpus_texts[c]) for c in cand])
            feats.append(expert_features([r[sel] for r in raw], dense, hubv, sparse, jws))

        # 4. Agreement weights over every query occurrence (each with its own ID for tie-breaks).
        refs_of = [[cids[c] for c in cand] for cand in pools]
        occ_refs = refs_of if len(uq_texts) == n else [refs_of[u] for u in q_inv]
        occ_feats = feats if len(uq_texts) == n else [feats[u] for u in q_inv]
        fused, weights, agreement = fuse(qids, occ_refs, occ_feats)
        self.last_weights, self.last_agreement = weights, agreement

        # 5. Top-1, margin and reranker probability per occurrence; then B2 over all queries.
        probs = np.full((len(raw), n), np.nan)
        for i in range(n):
            u = int(q_inv[i])
            if fused[i] is None:
                continue
            w, best, margin, ties = select_top(qids[i], occ_refs[i], fused[i])
            out["match_idx"][i] = int(pools[u][w])
            out["fused_score"][i] = best
            out["margin"][i] = margin
            out["top_tie_count"][i] = ties
            p = int(pair_sel[u][w])
            for m, r in enumerate(raw):
                probs[m, i] = float(r[p])
        out["reranker_probability"] = mean_probability(list(probs))
        conf = b2_confidence(out["margin"], out["reranker_probability"], out["top_tie_count"] > 1)
        out["score"] = np.where(out["match_idx"] >= 0, conf, np.nan)
        if details:
            out.update(pools=pools, query_inverse=q_inv, dense_pool=dense_pool, sparse_pool=sparse_pool, hub=hub,
                       reranker_scores=raw, pair_index=pair_sel, fused=fused, weights=weights, agreement=agreement)
        return out

    def match(
        self,
        query_texts: Sequence[str],
        corpus_texts: Sequence[str],
        show_progress: bool = True,
        batch_size: int = 50000,
        query_prompt: Optional[str] = None,
        query_ids: Optional[Sequence[str]] = None,
        corpus_ids: Optional[Sequence[str]] = None,
    ) -> List[Tuple[Optional[int], Optional[float]]]:
        """``(corpus_index, confidence)`` per query; ``(None, None)`` without candidates. See :meth:`link`."""
        res = self.link(query_texts, corpus_texts, query_ids=query_ids, corpus_ids=corpus_ids,
                        show_progress=show_progress, batch_size=batch_size, query_prompt=query_prompt)
        return [(None, None) if m < 0 else (int(m), float(s)) for m, s in zip(res["match_idx"], res["score"])]
