"""
Zero-shot record linkage.

``link`` matches a query table to a reference corpus with the four-expert
agreement-fusion core (see :mod:`zeroshot_linkage.core`). ``link_blocked`` adds
hierarchical blocking: it matches a coarse field first (e.g. state) and then
matches a detail field (e.g. county) only within the matched block. No labeled
training data is required for either.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Sequence

from .core import FusionMatcher, DEFAULT_EMBEDDING_MODEL, DEFAULT_RERANKER_MODELS


def _resolve_rerankers(reranker_models, reranker_model):
    """Honor the legacy single ``reranker_model`` kwarg if a caller passes it."""
    if reranker_model is not None:
        return [reranker_model]
    return list(reranker_models)


def _build_matcher(
    embedding_model,
    reranker_models,
    reranker_model,
    pool_size,
    retrieval_top_k,
    device,
    cache_dir,
) -> FusionMatcher:
    if retrieval_top_k is not None:
        pool_size = retrieval_top_k
    return FusionMatcher(
        embedding_model=embedding_model,
        reranker_models=_resolve_rerankers(reranker_models, reranker_model),
        pool_size=pool_size,
        device=device,
        cache_dir=cache_dir,
    )


def link(
    queries: pd.DataFrame,
    corpus: pd.DataFrame,
    column_query: str = None,
    column_corpus: Optional[str] = None,
    columns_query: Optional[list] = None,
    columns_corpus: Optional[list] = None,
    pool_size: int = 50,
    retrieval_top_k: Optional[int] = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    reranker_models: Sequence[str] = DEFAULT_RERANKER_MODELS,
    reranker_model: Optional[str] = None,
    show_progress: bool = True,
    cache_dir: Optional[str] = None,
    batch_size: int = 50000,
    device: Optional[str] = None,
) -> pd.DataFrame:
    """
    Link records from queries to corpus with four-expert agreement fusion.

    Candidates are retrieved by an embedding-plus-TF-IDF ensemble and then scored
    by four experts (a two-model reranker ensemble, a CSLS-corrected dense
    cosine, a sparse TF-IDF cosine, and Jaro-Winkler similarity). The experts are
    fused without labels by weighting each one by how often it agrees with the
    consensus pick. All models run locally; no API keys are needed.

    For multi-column matching, pass ``columns_query`` (and optionally
    ``columns_corpus``) instead of ``column_query``. Columns are concatenated
    with " | " separators before matching. Concatenation consistently
    outperforms blocking-based approaches across benchmarks (see Dasanaike 2026).

    Parameters
    ----------
    queries : pd.DataFrame
        The dataset to find matches for.
    corpus : pd.DataFrame
        The reference dataset to match against.
    column_query : str, optional
        Column in queries containing the text to match. Mutually exclusive with
        ``columns_query``.
    column_corpus : str, optional
        Column in corpus to match against. Defaults to ``column_query``.
    columns_query : list of str, optional
        Multiple query columns to concatenate. Mutually exclusive with
        ``column_query``.
    columns_corpus : list of str, optional
        Multiple corpus columns to concatenate. Defaults to ``columns_query``.
    pool_size : int
        Candidates retrieved per query from each of dense and sparse retrieval.
        Default: 50. ``retrieval_top_k`` is accepted as an alias.
    embedding_model : str
        Dense embedding model. Default: harrier-oss-v1-0.6b.
    reranker_models : sequence of str
        Cross-encoder rerankers forming the reranker expert. Default: Jina v2 and
        BGE v2-m3. Pass a single ``reranker_model`` to use just one.
    reranker_model : str, optional
        Legacy single-reranker override; takes precedence over ``reranker_models``.
    show_progress : bool
        Show progress bars. Default: True.
    cache_dir : str, optional
        Directory to download/cache models. Defaults to the HuggingFace cache.
    batch_size : int
        Number of corpus texts to embed at once. Default: 50,000.
    device : str, optional
        Device for inference ("cuda" or "cpu"). Defaults to GPU if available.

    Returns
    -------
    pd.DataFrame
        Columns: ``query_idx``, ``query_text``, ``match_idx``, ``match_text``,
        ``score`` (the fused confidence; higher is better).

    Example
    -------
    >>> import pandas as pd
    >>> from zeroshot_linkage import link
    >>> queries = pd.DataFrame({"name": ["John Smith", "Jane Doe"]})
    >>> corpus = pd.DataFrame({"name": ["J. Smith", "Jane M. Doe", "Bob Wilson"]})
    >>> results = link(queries, corpus, column_query="name")
    """
    if column_query is not None and columns_query is not None:
        raise ValueError("Specify either column_query or columns_query, not both.")
    if column_query is None and columns_query is None:
        raise ValueError("Specify either column_query or columns_query.")

    queries = queries.reset_index(drop=True)
    corpus = corpus.reset_index(drop=True)

    if columns_query is not None:
        if columns_corpus is None:
            columns_corpus = columns_query
        query_texts = queries[columns_query].astype(str).agg(" | ".join, axis=1).tolist()
        corpus_texts = corpus[columns_corpus].astype(str).agg(" | ".join, axis=1).tolist()
    else:
        if column_corpus is None:
            column_corpus = column_query
        query_texts = queries[column_query].astype(str).tolist()
        corpus_texts = corpus[column_corpus].astype(str).tolist()

    matcher = _build_matcher(
        embedding_model, reranker_models, reranker_model,
        pool_size, retrieval_top_k, device, cache_dir,
    )
    matches = matcher.match(
        query_texts, corpus_texts, show_progress=show_progress, batch_size=batch_size
    )

    results = []
    for query_idx, (match_idx, score) in enumerate(matches):
        results.append({
            "query_idx": query_idx,
            "query_text": query_texts[query_idx],
            "match_idx": match_idx,
            "match_text": corpus_texts[match_idx] if match_idx is not None else None,
            "score": score,
        })
    return pd.DataFrame(results)


def link_blocked(
    queries: pd.DataFrame,
    corpus: pd.DataFrame,
    blocking_query: str,
    detail_query: str,
    blocking_corpus: Optional[str] = None,
    detail_corpus: Optional[str] = None,
    pool_size: int = 50,
    retrieval_top_k: Optional[int] = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    reranker_models: Sequence[str] = DEFAULT_RERANKER_MODELS,
    reranker_model: Optional[str] = None,
    show_progress: bool = True,
    cache_dir: Optional[str] = None,
    batch_size: int = 50000,
    device: Optional[str] = None,
) -> pd.DataFrame:
    """
    Link records hierarchically: match a coarse block, then a detail within it.

    Example: match states first, then match counties only within the matched
    state. Both stages use the same four-expert agreement-fusion core.

    Parameters
    ----------
    queries, corpus : pd.DataFrame
        Query and reference datasets.
    blocking_query, detail_query : str
        Query columns for the coarse block (e.g. "state") and the detail (e.g.
        "county").
    blocking_corpus, detail_corpus : str, optional
        Corpus columns; default to the query column names.
    pool_size, retrieval_top_k, embedding_model, reranker_models, reranker_model,
    show_progress, cache_dir, batch_size, device
        As in :func:`link`.

    Returns
    -------
    pd.DataFrame
        Columns: ``query_idx``, ``query_block``, ``query_detail``, ``match_idx``,
        ``match_block``, ``match_detail``, ``block_score``, ``detail_score``.
    """
    if blocking_corpus is None:
        blocking_corpus = blocking_query
    if detail_corpus is None:
        detail_corpus = detail_query

    queries = queries.reset_index(drop=True)
    corpus = corpus.reset_index(drop=True)

    corpus_blocks = corpus[blocking_corpus].astype(str).tolist()
    corpus_details = corpus[detail_corpus].astype(str).tolist()

    matcher = _build_matcher(
        embedding_model, reranker_models, reranker_model,
        pool_size, retrieval_top_k, device, cache_dir,
    )

    # Stage 1: match each unique query block to the unique corpus blocks.
    unique_query_blocks = queries[blocking_query].astype(str).unique().tolist()
    unique_corpus_blocks = list(dict.fromkeys(corpus_blocks))
    if show_progress:
        print(f"Matching {len(unique_query_blocks)} unique blocking values...")
    block_matches = matcher.match(
        unique_query_blocks, unique_corpus_blocks, show_progress=show_progress
    )
    block_mapping = {}
    for qb, (m_idx, score) in zip(unique_query_blocks, block_matches):
        block_mapping[qb] = (
            unique_corpus_blocks[m_idx] if m_idx is not None else None,
            score,
        )

    # Pre-index corpus rows by block for the detail stage.
    block_to_rows = {}
    for idx, block in enumerate(corpus_blocks):
        block_to_rows.setdefault(block, []).append(idx)

    # Stage 2: detail matching within each matched block, grouped so a block's
    # corpus subset is embedded once and its queries are matched together.
    query_blocks = queries[blocking_query].astype(str).tolist()
    query_details = queries[detail_query].astype(str).tolist()

    detail_idx = [None] * len(queries)
    detail_score = [None] * len(queries)

    grouped = {}  # matched_corpus_block -> list of query row indices
    for q_idx, qb in enumerate(query_blocks):
        matched_block, _ = block_mapping.get(qb, (None, None))
        if matched_block is not None and matched_block in block_to_rows:
            grouped.setdefault(matched_block, []).append(q_idx)

    if show_progress:
        print(f"Matching details within {len(grouped)} blocks...")
    for matched_block, q_indices in grouped.items():
        rows = block_to_rows[matched_block]
        sub_corpus = [corpus_details[r] for r in rows]
        sub_queries = [query_details[q] for q in q_indices]
        sub_matches = matcher.match(sub_queries, sub_corpus, show_progress=False)
        for q_idx, (local_idx, score) in zip(q_indices, sub_matches):
            if local_idx is not None:
                detail_idx[q_idx] = rows[local_idx]
                detail_score[q_idx] = score

    results = []
    for q_idx in range(len(queries)):
        qb = query_blocks[q_idx]
        matched_block, block_score = block_mapping.get(qb, (None, None))
        m_idx = detail_idx[q_idx]
        results.append({
            "query_idx": q_idx,
            "query_block": qb,
            "query_detail": query_details[q_idx],
            "match_idx": m_idx,
            "match_block": matched_block,
            "match_detail": corpus_details[m_idx] if m_idx is not None else None,
            "block_score": block_score,
            "detail_score": detail_score[q_idx],
        })
    return pd.DataFrame(results)
