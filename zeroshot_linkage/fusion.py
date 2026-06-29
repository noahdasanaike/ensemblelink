"""
Label-free agreement fusion over experts.

This module holds the decision rule at the heart of EnsembleLink. Several
independent experts each score the candidate pool for a query; the experts are
combined without any labeled data by weighting each one by how often its top
pick agrees with the consensus pick across all queries. Experts that agree with
the crowd are trusted; experts that disagree are discounted. The weighting is
squared so that a small advantage in agreement translates into a larger
advantage in weight.

The primitives here are deliberately small and stateless so that they can be
reused by the record-linkage core, the R backend, and any downstream
experiments.
"""

import numpy as np
from collections import Counter


def zscore(a):
    """z-normalize a vector. Returns zeros when the input has no spread.

    Z-normalization puts every expert on a common scale before fusion, so that
    a reranker logit and a cosine similarity contribute comparably.
    """
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        return a
    s = a.std()
    if s > 1e-9:
        return (a - a.mean()) / s
    return np.zeros_like(a)


def agreement_weights(expert_mats):
    """Squared-agreement weights for a list of per-query expert matrices.

    Parameters
    ----------
    expert_mats : list of np.ndarray
        One matrix per query, each of shape (n_candidates, n_experts). Column
        ``e`` holds expert ``e``'s (already z-normalized) scores for that
        query's candidate pool.

    Returns
    -------
    np.ndarray
        A length-``n_experts`` weight vector. For each query, every expert votes
        for its argmax candidate; the consensus is the most common vote. An
        expert's weight is the squared fraction of queries on which it agrees
        with the consensus. No labels are used.
    """
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

    weights = (agree / len(mats)) ** 2
    if weights.sum() <= 0:
        # Degenerate case (e.g., single candidate everywhere): fall back to a
        # uniform vote so fusion still returns a sensible argmax.
        weights = np.ones(n_experts)
    return weights


def csls_hubness(corpus_emb, query_emb, pools, k=10):
    """Per-candidate hubness term for the CSLS correction.

    Dense nearest-neighbour retrieval suffers from hubs: a handful of corpus
    points sit close to many queries and get retrieved spuriously. CSLS
    (Cross-domain Similarity Local Scaling) subtracts, from each candidate's
    cosine, a measure of how close that candidate sits to the query set as a
    whole. Here that measure is the mean of the candidate's top-``k`` cosines
    over all queries.

    Parameters
    ----------
    corpus_emb : np.ndarray
        Normalized corpus embeddings, shape (n_corpus, dim).
    query_emb : np.ndarray
        Normalized query embeddings, shape (n_queries, dim).
    pools : list of list of int
        Candidate corpus indices per query.
    k : int
        Neighbourhood size for the hubness average.

    Returns
    -------
    dict
        Maps each pooled corpus index to its scalar hubness term.
    """
    uniq = sorted({int(c) for cand in pools for c in cand})
    if not uniq:
        return {}
    sub = corpus_emb[uniq] @ query_emb.T  # (n_unique_candidates, n_queries)
    kk = min(k, sub.shape[1])
    return {
        uniq[i]: float(np.mean(np.sort(sub[i])[-kk:]))
        for i in range(len(uniq))
    }
