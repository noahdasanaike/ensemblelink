"""
Label-free agreement fusion, top-1 selection and the B2 confidence.

These are the decision rules of EnsembleLink, written as the paper's benchmark
engine computes them (``ensemblelink_core.py`` and ``evaluate_linkage.py`` of the
replication archive), so that the package reproduces its numbers:

* every expert is z-scored within a query's candidate pool (a constant expert
  becomes all zeros); the reranker expert is the z-score of the sum of the
  rerankers' z-scores;
* each expert votes for its top candidate; an expert with no spread over the
  pool abstains; the consensus is the modal vote;
* an expert's weight is the squared share of nonempty query pools on which it
  votes and agrees with the consensus, normalized to sum to one;
* ties (an expert's argmax, the modal vote and the final top-1) are broken by
  the smallest SHA-256 of ``"<query id>|<record id>"``;
* the returned confidence is rule B2: the mean of the percentile ranks, over
  all queries linked in the call, of the fused top-minus-runner-up margin and
  of the proposed candidate's mean 0-1 reranker probability. A query whose top
  fused score is tied gets confidence 0.

No labeled data enter any step.
"""

import hashlib
from typing import List, Optional, Sequence

import numpy as np

MULTIFIELD_PROMPT = "Instruct: Retrieve the record that refers to the same entity\nQuery: "


def lexical_text(text: str) -> str:
    """Text read by the character-level experts: transliterated to Latin letters with Unidecode.

    Transliteration lets character n-grams and Jaro-Winkler compare names written
    in different scripts (Moskva / Moscow); for Latin-script text it only removes
    diacritics.
    """
    from unidecode import unidecode

    return unidecode(text)


def zscore(x) -> np.ndarray:
    """Population z-score; zeros when the input has (almost) no spread."""
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite expert score")
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-9 else np.zeros_like(x)


def _digest(q: str, r: str) -> bytes:
    return hashlib.sha256((q + "|" + r).encode()).digest()


def tied_pick(q: str, refs: Sequence[str], values) -> int:
    """Index of the maximum of ``values``; ties go to the smallest SHA-256 of ``q|ref``."""
    values = np.asarray(values)
    tied = np.flatnonzero(values == np.max(values))
    if len(tied) == 1:
        return int(tied[0])
    return int(min(tied, key=lambda i: _digest(q, refs[i])))


def top_k(scores, k: int) -> np.ndarray:
    """Indices of the ``k`` largest scores, in descending order.

    Scores equal to the k-th largest are kept in index (corpus-row) order, and
    the result is sorted by score, then index. With the corpus in record-ID
    order this is the benchmark's rule (ties broken by reference ID).
    """
    scores = np.asarray(scores)
    k = min(k, len(scores))
    if k <= 0:
        return np.array([], dtype=np.int64)
    threshold = np.partition(scores, len(scores) - k)[len(scores) - k]
    above = np.flatnonzero(scores > threshold)
    equal = np.flatnonzero(scores == threshold)
    chosen = np.r_[above, equal[: k - len(above)]]
    return chosen[np.lexsort((chosen, -scores[chosen]))]


def expert_features(reranker_scores: Sequence[np.ndarray], dense, hub, sparse, jw) -> np.ndarray:
    """The four z-scored experts of one query pool, shape (n_candidates, 4).

    reranker expert z(sum_m z(reranker_m)); CSLS dense z(2 cos - hub); sparse z(TF-IDF cosine);
    lexical z(Jaro-Winkler).
    """
    rer = zscore(reranker_scores[0])
    for s in reranker_scores[1:]:
        rer = rer + zscore(s)
    dense = np.asarray(dense, dtype=float)
    hub = np.asarray(hub, dtype=float)
    return np.column_stack([zscore(rer), zscore(2 * dense - hub), zscore(sparse), zscore(jw)])


def expert_votes(q: str, refs: Sequence[str], f: np.ndarray):
    """Each expert votes for its top candidate; an expert with no spread over the pool abstains (-1).

    Without abstention, experts that are constant on a pool (for example lexical
    scorers comparing strings in different scripts) would all fall to the same
    tie-break candidate, agree with each other, and outvote the informative experts.
    """
    selected = np.array([tied_pick(q, refs, f[:, i]) if np.ptp(f[:, i]) > 0 else -1 for i in range(f.shape[1])])
    voting = selected[selected >= 0]
    if len(voting) == 0:
        return selected, -1
    votes = np.bincount(voting, minlength=len(refs))
    return selected, tied_pick(q, refs, votes)


def agreement_weights(picks: np.ndarray, consensus: np.ndarray) -> np.ndarray:
    """Squared share of nonempty pools on which each expert votes and agrees with the consensus, summing to one."""
    picks = np.asarray(picks)
    consensus = np.asarray(consensus)
    agreement = np.mean((picks == consensus[:, None]) & (picks >= 0), axis=0)
    weights = agreement ** 2
    if weights.sum() == 0:
        # Degenerate (for example one candidate per query): fall back to equal weights.
        weights = np.ones_like(weights)
    return weights / weights.sum()


def fuse(query_ids: Sequence[str], refs: Sequence[Sequence[str]], features: Sequence[Optional[np.ndarray]]):
    """Fit the agreement weights over all nonempty pools and return (fused scores per query, weights, agreement)."""
    picks, consensus = [], []
    for q, r, f in zip(query_ids, refs, features):
        if f is None or len(f) == 0:
            continue
        s, m = expert_votes(q, r, f)
        picks.append(s)
        consensus.append(m)
    if not picks:
        return [None] * len(features), None, None
    picks = np.asarray(picks)
    consensus = np.asarray(consensus)
    weights = agreement_weights(picks, consensus)
    agreement = np.mean((picks == consensus[:, None]) & (picks >= 0), axis=0)
    scores = [None if f is None or len(f) == 0 else f @ weights for f in features]
    return scores, weights, agreement


def select_top(q: str, refs: Sequence[str], fused: np.ndarray):
    """Top-1 of a pool: (position, top fused score, margin, number tied at the top).

    Ties go to the smallest SHA-256 hex digest of ``q|ref``; the margin is 0 for a tie
    at the top and NaN for a single-candidate pool.
    """
    fused = np.asarray(fused, dtype=float)
    best = float(fused.max())
    tied = np.flatnonzero(fused == best)
    if len(tied) == 1:
        winner = int(tied[0])
    else:
        winner = int(min(tied, key=lambda i: hashlib.sha256((q + "|" + refs[i]).encode()).hexdigest()))
    lower = fused[fused < best]
    if len(tied) > 1:
        margin = 0.0
    elif len(lower):
        margin = best - float(lower.max())
    else:
        margin = float("nan")
    return winner, best, margin, len(tied)


def b2_confidence(margin, probability, tie) -> np.ndarray:
    """Rule B2: (rank(margin) + rank(p)) / (2n), average ranks over the n queries of the call.

    ``margin`` NaN counts as 0; ``p`` of a tied or candidate-less query is 0 before
    ranking; tied queries get confidence 0.
    """
    from scipy.stats import rankdata

    margin = np.nan_to_num(np.asarray(margin, dtype=np.float64), nan=0.0)
    tie = np.asarray(tie, dtype=bool)
    if not (len(margin) == len(tie) == len(probability)):
        raise ValueError("Confidence inputs differ in length")
    n = len(margin)
    b1 = np.where(tie, 0.0, np.nan_to_num(np.asarray(probability, dtype=np.float64), nan=0.0))
    return np.where(tie, 0.0, (rankdata(margin) / n + rankdata(b1) / n) / 2) if n else b1


def mean_probability(probabilities: List[np.ndarray]) -> np.ndarray:
    """Mean 0-1 reranker score over the rerankers (float32 scores averaged in float64)."""
    return np.mean([np.asarray(p, dtype=np.float64) for p in probabilities], axis=0)
