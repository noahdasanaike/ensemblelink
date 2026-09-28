"""
Zero-shot record linkage.

``link`` matches a query table to a reference corpus with the four-expert
agreement-fusion core (see :mod:`zeroshot_linkage.core`). ``link_blocked`` adds
hierarchical blocking: it matches a coarse field first (e.g. state) and then
matches a detail field (e.g. county) only within the matched block. No labeled
training data is required for either.
"""

from typing import List, Optional, Sequence

import pandas as pd

from .core import FusionMatcher, DEFAULT_EMBEDDING_MODEL, DEFAULT_RERANKER_MODELS
from .fusion import MULTIFIELD_PROMPT
from .reranker import DEFAULT_RERANKER_BATCH, DEFAULT_RERANKER_MAX_LENGTH
from .retrieval import DEFAULT_EMBED_MAX_LENGTH


def record_texts(frame: pd.DataFrame, columns: Sequence[str], labels: Sequence[str]) -> List[str]:
    """Record text under the field-count rule.

    Records with two or more fields are written as ``label=value`` for each
    nonmissing field, joined by " | ", so that the models can tell which value
    is which (for example a capacity versus a speed). A single field is written
    as its value alone: a label would add only identical text to every record,
    which the character-level experts would score as agreement. Labels are the
    query-side column names, paired with corpus columns by position. Missing
    values are omitted rather than written as "nan" or "None". Store whole
    numbers (years) as integers (e.g. pandas ``Int64``) so they read "1980", not
    "1980.0".
    """
    texts = []
    multi = len(columns) > 1
    for row in frame[list(columns)].itertuples(index=False, name=None):
        parts = []
        for label, value in zip(labels, row):
            if value is None or (not isinstance(value, str) and pd.isna(value)):
                continue
            value = str(value).strip()
            if value:
                parts.append(f"{label}={value}" if multi else value)
        texts.append(" | ".join(parts))
    return texts


def _resolve_rerankers(reranker_models, reranker_model):
    """Honor the legacy single ``reranker_model`` kwarg if a caller passes it."""
    if reranker_model is not None:
        return [reranker_model]
    return list(reranker_models)


def _build_matcher(embedding_model, reranker_models, reranker_model, pool_size, retrieval_top_k, device,
                   cache_dir, exact=True, index_cache=None, **kw) -> FusionMatcher:
    if retrieval_top_k is not None:
        pool_size = retrieval_top_k
    return FusionMatcher(
        embedding_model=embedding_model,
        reranker_models=_resolve_rerankers(reranker_models, reranker_model),
        pool_size=pool_size, device=device, cache_dir=cache_dir, exact=exact, index_cache=index_cache, **kw,
    )


def link(
    queries: pd.DataFrame,
    corpus: pd.DataFrame,
    column_query: str = None,
    column_corpus: Optional[str] = None,
    columns_query: Optional[list] = None,
    columns_corpus: Optional[list] = None,
    id_query: Optional[str] = None,
    id_corpus: Optional[str] = None,
    pool_size: int = 30,
    retrieval_top_k: Optional[int] = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    reranker_models: Sequence[str] = DEFAULT_RERANKER_MODELS,
    reranker_model: Optional[str] = None,
    show_progress: bool = True,
    cache_dir: Optional[str] = None,
    batch_size: int = 50000,
    device: Optional[str] = None,
    exact: bool = True,
    index_cache: Optional[str] = None,
    max_length: int = DEFAULT_RERANKER_MAX_LENGTH,
    embed_max_length: int = DEFAULT_EMBED_MAX_LENGTH,
    embed_batch_size: Optional[int] = None,
    reranker_batch_size: int = DEFAULT_RERANKER_BATCH,
    dtype: str = "auto",
) -> pd.DataFrame:
    """
    Link records from queries to corpus with four-expert agreement fusion.

    Candidates are retrieved by exact dense search plus character TF-IDF and
    scored by four experts (a two-model reranker ensemble, a CSLS-corrected
    dense cosine, a sparse TF-IDF cosine and Jaro-Winkler similarity), fused
    without labels by weighting each expert by how often it agrees with the
    consensus pick. All models run locally; no API keys are needed.

    For multi-column matching, pass ``columns_query`` (and optionally
    ``columns_corpus``). Columns are written under the field-count rule
    (``column=value`` for each nonmissing column, joined by " | "; see
    ``record_texts``), and multi-field queries get the embedding instruction
    "Retrieve the record that refers to the same entity".

    Parameters
    ----------
    queries, corpus : pd.DataFrame
        The records to link and the reference records.
    column_query, column_corpus : str
        Single text column (``column_corpus`` defaults to ``column_query``).
    columns_query, columns_corpus : list of str
        Several columns, paired by position (``columns_corpus`` defaults to ``columns_query``).
    id_query, id_corpus : str, optional
        Columns of unique record IDs, used only to break exact ties (smallest
        SHA-256 of "query id|record id"). Default: row positions.
    pool_size : int
        Candidates retrieved per query from each of dense and sparse retrieval (default 30).
        ``retrieval_top_k`` is an alias.
    embedding_model : str
        Dense embedding model (default harrier-oss-v1-0.6b, MIT).
    reranker_models : sequence of str
        Rerankers forming the reranker expert. Default Jina Reranker v2
        (CC-BY-NC-4.0, non-commercial) and BGE Reranker v2-m3 (Apache-2.0).
        ``COMMERCIAL_RERANKER_MODELS`` (BGE v2-m3 and zerank-2) is licensed for
        commercial use (see README, Licenses). ``reranker_model`` (legacy) uses a single model.
    show_progress : bool
        Show progress bars.
    cache_dir : str, optional
        Model cache directory (default: the Hugging Face cache).
    batch_size : int
        Texts per embedding call in fast mode (exact mode uses 10,000).
    device : str, optional
        "cuda" or "cpu" (default: GPU if available).
    exact : bool
        True (default): the models see the inputs and batches of the paper's
        benchmark, whose results the package reproduces. False: each distinct
        text and (query, candidate) pair is scored once, in length-sorted
        batches; faster, with noise-level score differences (README, Speed).
    index_cache : str, optional
        Directory for an on-disk cache of the corpus index; a later call with the
        same corpus and settings loads it instead of re-embedding.
    max_length, embed_max_length : int
        Token ceilings for reranker pairs (1024) and embedded records (512).
    embed_batch_size, reranker_batch_size : int
        Batch sizes (embedding 256 in exact mode, adaptive in fast mode; reranker 128).
    dtype : str
        Model precision: "auto" = bfloat16 on GPU (the paper's), float32 on CPU.

    Returns
    -------
    pd.DataFrame
        One row per query: ``query_idx``, ``query_text``, ``match_idx``,
        ``match_text``, ``score``, ``margin``, ``reranker_probability``,
        ``fused_score``. ``score`` is the B2 confidence in (0, 1]: the mean of the
        percentile ranks, among the queries of this call, of the fused
        top-minus-runner-up margin and of the match's mean reranker probability
        (0 when the top fused score is tied). It ranks queries within one call; it
        is not a probability, and a threshold chosen on one call applies to another
        only if the query sets are alike (README, Confidence).
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
        if len(columns_query) != len(columns_corpus):
            raise ValueError("columns_query and columns_corpus must have the same length.")
        query_texts = record_texts(queries, columns_query, columns_query)
        corpus_texts = record_texts(corpus, columns_corpus, columns_query)
    else:
        if column_corpus is None:
            column_corpus = column_query
        query_texts = record_texts(queries, [column_query], [column_query])
        corpus_texts = record_texts(corpus, [column_corpus], [column_query])

    matcher = _build_matcher(
        embedding_model, reranker_models, reranker_model, pool_size, retrieval_top_k, device, cache_dir,
        exact, index_cache, max_length=max_length, embed_max_length=embed_max_length,
        embed_batch_size=embed_batch_size, reranker_batch_size=reranker_batch_size, dtype=dtype,
    )
    # Records with two or more fields receive an embedding instruction; a single field does not.
    multi = columns_query is not None and len(columns_query) > 1
    res = matcher.link(
        query_texts, corpus_texts,
        query_ids=None if id_query is None else queries[id_query].astype(str).tolist(),
        corpus_ids=None if id_corpus is None else corpus[id_corpus].astype(str).tolist(),
        show_progress=show_progress, batch_size=batch_size,
        query_prompt=MULTIFIELD_PROMPT if multi else None,
    )
    idx = res["match_idx"]
    return pd.DataFrame({
        "query_idx": range(len(query_texts)),
        "query_text": query_texts,
        "match_idx": pd.array([None if m < 0 else int(m) for m in idx], dtype="Int64"),
        "match_text": [corpus_texts[m] if m >= 0 else None for m in idx],
        "score": res["score"],
        "margin": res["margin"],
        "reranker_probability": res["reranker_probability"],
        "fused_score": res["fused_score"],
    })


def link_blocked(
    queries: pd.DataFrame,
    corpus: pd.DataFrame,
    blocking_query: str,
    detail_query: str,
    blocking_corpus: Optional[str] = None,
    detail_corpus: Optional[str] = None,
    pool_size: int = 30,
    retrieval_top_k: Optional[int] = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    reranker_models: Sequence[str] = DEFAULT_RERANKER_MODELS,
    reranker_model: Optional[str] = None,
    show_progress: bool = True,
    cache_dir: Optional[str] = None,
    batch_size: int = 50000,
    device: Optional[str] = None,
    exact: bool = True,
    index_cache: Optional[str] = None,
) -> pd.DataFrame:
    """
    Link records hierarchically: match a coarse block, then a detail within it.

    Example: match states first, then counties only within the matched state.
    Both stages use the same four-expert core. ``block_score`` and
    ``detail_score`` are B2 confidences, ranked within the block stage and
    within each matched block's detail call respectively.

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

    matcher = _build_matcher(embedding_model, reranker_models, reranker_model, pool_size, retrieval_top_k,
                             device, cache_dir, exact, index_cache)

    # Stage 1: each unique query block against the unique corpus blocks.
    unique_query_blocks = queries[blocking_query].astype(str).unique().tolist()
    unique_corpus_blocks = list(dict.fromkeys(corpus_blocks))
    if show_progress:
        print(f"Matching {len(unique_query_blocks)} unique blocking values...")
    block_matches = matcher.match(unique_query_blocks, unique_corpus_blocks, show_progress=show_progress)
    block_mapping = {qb: (unique_corpus_blocks[m] if m is not None else None, s)
                     for qb, (m, s) in zip(unique_query_blocks, block_matches)}

    block_to_rows = {}
    for idx, block in enumerate(corpus_blocks):
        block_to_rows.setdefault(block, []).append(idx)

    # Stage 2: detail matching within each matched block.
    query_blocks = queries[blocking_query].astype(str).tolist()
    query_details = queries[detail_query].astype(str).tolist()
    detail_idx = [None] * len(queries)
    detail_score = [None] * len(queries)
    grouped = {}
    for q_idx, qb in enumerate(query_blocks):
        matched_block, _ = block_mapping.get(qb, (None, None))
        if matched_block is not None and matched_block in block_to_rows:
            grouped.setdefault(matched_block, []).append(q_idx)
    if show_progress:
        print(f"Matching details within {len(grouped)} blocks...")
    for matched_block, q_indices in grouped.items():
        rows = block_to_rows[matched_block]
        sub_matches = matcher.match([query_details[q] for q in q_indices], [corpus_details[r] for r in rows],
                                    show_progress=False)
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
